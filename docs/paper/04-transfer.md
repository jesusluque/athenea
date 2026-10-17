<!-- Copyright (c) 2026 jesus luque. -->

# 4. Light transport baked into Gaussians (transfer) and relighting

A cloud converted from a mesh can carry its light in two ways. A *radiance bake* stores, per Gaussian, the light that
arrived under the stage's own illumination; it is exact for that illumination and wrong under any other. A *transfer*
stores instead how each Gaussian's surroundings would carry an arbitrary environment to it, and a frame combines that
geometry with whatever sky and lights it has. This section describes the core of the work, the baker: the transfer that the scene's
path tracer bakes into every Gaussian, and the relighting that evaluates it at draw time, in its TX form (a transfer that holds under any sky and any light), the reflection model built on top of it, the
per-pixel treatment of sharp lobes, and what each piece was measured to buy against a path trace of the source mesh.
Everything described runs on the GPU: the bake is the scene's own path tracer in a transfer mode, and the evaluation is
the splat projection and blend kernels (`shaders/athenea/splat/splat_relight.slang`, `splat_project.slang`,
`splat_blend.slang`).

## 4.1 Notation

A Gaussian has a position $\mathbf{p}$, a unit normal $\mathbf{n}$ (its shortest axis, or a stored shading normal,
turned towards the eye), an albedo $\rho$, a metalness $m$, a perceptual roughness $r$ with GGX parameter
$\alpha = r^2$, and optionally a clear coat (weight $w_c$, roughness $r_c$, index $\eta_c$), a sheen and a
transmission. $\boldsymbol{\omega}_o$ is the direction to the eye and $\mathbf{m} = \operatorname{reflect}(-\boldsymbol{\omega}_o,
\mathbf{n})$ the mirror direction. $Y_k$ are the real spherical harmonics, indexed $k = l^2 + l + m$; degree 2 has 9
coefficients and degree 3 has 16. A sky has radiance coefficients $L_k$ (rgb), a sun of direction $\mathbf{s}$ and
irradiance $E_\odot$ extracted from it, and a prefiltered map $\tilde L(\boldsymbol{\omega}, r)$.

## 4.2 What a Gaussian stores

The transfer is one run of values per Gaussian, f16 on the device. There is no mode token: the count is the layout
(`common/transfer_layout.slang`) (Figure 10).

| values | layout | origin |
|---|---|---|
| 9 | degree 2, direct | first transfer |
| 36 | degree 2, direct + indirect (27) | first transfer |
| 84 | degree 2, both + reflected field (48) | TX at `--transfer-degree 2` |
| 16 | degree 3, direct | |
| 64 | degree 3, direct + indirect (48) | |
| 112 | degree 3, both + reflected field (48) | **TX default** |
| 10 | two zonal lobes in the Gaussian's frame | skinned clouds |

![Figure 10. What the bake stores per Gaussian and how the draw combines it with the environment: the body from the direct and indirect SH, polish and clear coat from the reflected field and the open-direction cells, and the sharp lobe evaluated per pixel.](figures/fig-transfer.svg)

Beside it:

- **Open directions** (`primvars:athenea:splat:shadowBits`): 2 words (64 bits, an 8 x 8 octahedral grid; first
  transfer), 8 words (16 x 16 cells over the whole sphere; TX default) or 32 words (32 x 32). A bit is set where the
  bake's ray in that cell left the scene.
- **Curvature** (`primvars:athenea:splat:curvature`): the shape operator in the Gaussian's two in-plane axes,
  $(S_{uu}, S_{uv}, S_{vv})$, three floats, written with `--transfer`.
- **Material roughness** (`primvars:athenea:splat:materialRoughness`): the material's own roughness before the bake's
  widening (section 4.9), -1 under a texture map. On the device it rides in five bits (27-31) of the pbr word, flagged
  by bit 26.

The direct half is a scalar per coefficient (geometry alone); the indirect half is rgb, coloured by what light bounced
off; the reflected field is rgb. The TX layout costs, per Gaussian:

| | first transfer | TX |
|---|---|---|
| direct | 9 | 16 |
| indirect | 27 | 48 |
| reflected field | -- | 48 |
| visibility | 2 ints (64 bits) | 8 ints (256 bits) |
| file, bytes | 152 | 480 |
| device, bytes | 80 | 256 |

For the whole Corvette (about 10 M Gaussians at the time) that is 4.8 GB of file and 2.6 GB of device for the transfer
alone; the codecs that reduce its storage and bandwidth (a clustered PCA of the transfer, a Hamming k-means codebook of the shadow
bits) are described in Section 7, where they serve one application, a WebGPU player.

## 4.3 The environment at draw time

`technique::Environment` prepares each dome in three forms, all buffers read through one module
(`light/environment.slang`) by both the rasteriser and the tracer.

**Spherical harmonics of radiance.** `env_project` sums the texels of one level of the dome's mip chain, each weighed by
the lat-long solid angle $2\pi^2 \sin\theta\,\Delta u\,\Delta v$ (the measure the dome's importance sampling uses):

$$L_k = \sum_{\text{texels}} L(\boldsymbol{\omega}_t)\, Y_k(\boldsymbol{\omega}_t)\, 2\pi^2 \sin\theta_t\, \Delta u\, \Delta v .$$

Sixteen coefficients are projected (degree 3); the irradiance of an untransferred body reads the first nine, convolved
at read time with the clamped-cosine zonal weights $\{\pi, 2\pi/3, \pi/4\}$. A constant sky projects back to
$\pi L$ of irradiance at every normal: 6.2841 against 6.2832 (0.089 %) over 64 directions.

**The sun, taken out.** A disc a thousand times brighter than the sky is not a degree-2 function; projected, it smears
into a lobe sixty degrees wide. `env_sun` finds the brightest region whose light is as concentrated as a disc of its
solid angle (light-weighted mean square angle about its centre at most twice a disc's, $\theta_{\text{cone}}^2/2$),
takes its texels out of both the harmonics and the prefilter's source, and returns a direction $\mathbf{s}$, an
irradiance $E_\odot$ and a half angle. The irradiance is summed over exactly the texels the harmonics skip, at the same
level and with the same measure, so what leaves the coefficients arrives as a light (closed form 0.1119 against 0.1113).

**The prefiltered map.** The sky without its sun, convolved with a GGX lobe into an octahedral mip chain of eight
levels, level $k$ standing for roughness $(k/7)^2$ (0, 0.020, 0.082, 0.184, 0.327, 0.510, 0.735, 1). The base side
matches the dome's own texel, $S = \sqrt{2WH/\pi}$, rounded down to a power of two in $[256, 2048]$. The prefilter uses
filtered importance sampling (Krivanek and Colbert): each GGX sample reads the source one level coarser than its own
solid angle asks, never narrower than the texel it writes, with the sequence turned by a per-texel hash, 64 samples at
level 1 doubling to 1024. The source's lat-long mips weigh each row by $\cos\theta_0 - \cos\theta_1$, so every level
keeps the sky's solid-angle mean. Without these, a 1.5-degree sun in a rough level was a scatter of texel-sized blobs
(a paint ball's 4-16 px band at 2.13 times the path trace).

## 4.4 The bake

### 4.4.1 The transfer integral and its estimator

The transfer is the projection of visibility times the clamped cosine, the albedo taken as one:

$$T_k = \frac{1}{\pi} \int_{S^2} V(\boldsymbol{\omega})\, \max(\mathbf{n}\cdot\boldsymbol{\omega}, 0)\, Y_k(\boldsymbol{\omega})\, d\boldsymbol{\omega},$$

and a frame recombines it with the sky as $\langle T, L \rangle = \sum_k T_k L_k$. It is gathered by a third variant of
the path-traced kernel, `kTransfer`, beside the radiance bake and the camera path. At a Gaussian's first vertex the
material is replaced by the sampled direction itself: directions are stratified (a jittered $\sqrt{N}\times\sqrt{N}$
grid) and uniform over the hemisphere, density $1/2\pi$, against an estimator that wants $\cos\theta/\pi$, so an
escaping sample is worth $2\cos\theta\, Y_k(\boldsymbol{\omega})$ and a blocked one nothing. Uniform rather than cosine
sampling, and over the hemisphere measured as a sphere with the far half empty, because the harmonics are orthonormal
only over the sphere; fitted coefficient by coefficient over a hemisphere they overshoot (the pawn came back ten times
too bright). The accumulation *is* the projection; no fit is solved.

The same paths give two halves at no extra ray: a path that escapes right after the first vertex contributes to the
direct half (scalar); one that escapes after bouncing carries the colour of what it met and contributes to the indirect
half $T^{\text{ind}}_k$ (rgb). Checked against closed forms: an unoccluded point transfers the clamped cosine,
$(0.2820, 0.3258, 0.1581)$ against the exact $(0.2821, 0.3257, 0.1577)$ for bands 0-2, and a constant sky of 0.7
reads 0.6998. A corner of floor and wall, transfer recombined against a radiance bake of the same points:

| | transfer | radiance bake | worst point |
|---|---|---|---|
| no wall, 1 bounce | 0.6000 | 0.6000 | 0.0 % |
| black wall, 1 bounce | 0.3091 | 0.3039 | 2.5 % |
| white wall, 3 bounces | 0.5095 | 0.5325 | 4.3 % |

### 4.4.2 Open directions and the direct half by quadrature

At the first sample of each point TX traces one any-hit ray per cell of a $S_c \times S_c$ octahedral grid
($S_c = 16$ or 32) over the *whole* sphere: the far half too, starting from the far side of the surface, so a direction
behind a solid meets the solid and one behind a sheet leaves. A cell's bit is set where at least half the light gets
through: the occlusion ray passes up to eight surfaces whose material transmits, weighed by transmission times colour
luminance less the Fresnel at that angle (a ninth closes the way). The cells are traced from four origins over the
Gaussian's footprint (cell $c$ from origin $(c + \lfloor c/S_c \rfloor) \bmod 4$: the centre and three points brought
onto the surface within the Gaussian's disc), so a groove narrower than a Gaussian no longer reads as a dashed line of
whole shadows.

The same rays then give the direct half by quadrature (no grain), each cell weighted by its octahedral solid angle:

$$T_k^{\text{dir}} = \sum_{c} \tau_c\, \frac{\max(\mathbf{n}\cdot\boldsymbol{\omega}_c,0)}{\pi}\, \Delta\Omega_c\, Y_k(\boldsymbol{\omega}_c),
\qquad \Delta\Omega_c = \frac{4}{S_c^2}\,\lVert\boldsymbol{\omega}_c\rVert_1^3,$$

with $\tau_c$ the fraction transmitted along the cell's ray. Words are written four at a time as each 128 cells are
traced, so the kernel holds no array the size of the grid.

### 4.4.3 The reflected field

A reflection needs the bounced light *by direction*, not integrated. For every path that leaves the Gaussian, meets the
scene and later escapes, the throughput past the first vertex is the radiance that arrives along its first direction
$\boldsymbol{\omega}_1$ under a white sky of radiance one. It is projected onto degree 3 of $\boldsymbol{\omega}_1$,
uniformly over the domain it was drawn from:

$$F_j \approx \frac{1}{N} \sum_{i:\ \text{escaped after a bounce}} \Omega_{\text{meas}}\, \frac{\beta_i}{w_i}\, Y_j(\boldsymbol{\omega}_{1,i}),$$

with $\beta_i$ the throughput, $w_i$ the first-vertex weight and $\Omega_{\text{meas}} = 2\pi$ (or $4\pi$ over the
sphere). That is 16 rgb coefficients, `primvars:athenea:splat:transferReflected`. It costs no ray. Because $F$ is zero
along open directions, sixteen harmonics read a closed direction beside an open one at a fraction of what stands there
(the paint's closed part was 0.0014 of 0.115). So the bake also projects the indicator of closed directions $C_j$ over
the same samples, and the conversion divides the two, a normalised convolution giving the mean radiance of the closed
directions about each direction, falling back to the closed directions' mean where their share is under 0.1 (blended
up to 0.4), and re-projects the result over a 16 x 16 grid by cell solid angle (`m2sFieldOverClosed`).

For a transmitting Gaussian the first direction is drawn over the whole sphere (`bakeSphereDirection`: the stratum's
height stretched to $[-1,1]$ and the lower half mirrored through the tangent plane). Front samples keep the transfer's
estimator ($4\cos\theta$ at density $1/4\pi$); back samples feed only the field, after being bent into a solid
dielectric by its index (`refractInto`), so the field along $-\boldsymbol{\omega}_o$ holds what the eye's ray finds
through both faces. Unbent, a glass ball validated at half the path trace (0.265 against 0.541 under a white dome).

### 4.4.4 Slices, filtering, and the split bake

The whole TX Corvette asked for its bake's answer at once: 35 float4 per Gaussian, 7.8 GB, with 5 GB free.
`Converter::transfer` therefore bakes in slices whose answer is at most 1.5 GB (never under the bake's own batch,
$2^{19}$), and at most $2^{20}$ Gaussians where the filter runs. Each slice's rays are copied on the device, baked
(`StageRenderer::bakePointsOnDevice` in transfer mode), its bounced halves filtered, written by `m2sTransferInto` and
read back into the file's arrays before the next. The radiance bake uses the related `StageRenderer::bakeSplitOnDevice`,
which keeps the direct and indirect halves as separate sums so that the indirect one can be filtered before
`combineBake` fits the two. The transfer follows the same principle: its indirect half and its field (the noisiest
values, at 64 paths per Gaussian in the Corvette's bakes) go through the splat bake filter, an a-trous filter at
`--bake-filter` iterations (3), whose weights are position on the tangent plane, facing and prim id; the direct half,
which holds the shadows, is never filtered.

### 4.4.5 Zonal lobes for skinned clouds

Nine harmonics are in the world; a skeleton turns the Gaussian under them. Rotating them costs a 9 x 9 per Gaussian. A
zonal lobe rotates by rotating its axis, so a skinned cloud keeps two lobes, each the octahedral $(u, v)$ of its axis in
the Gaussian's own frame and zonal coefficients $z_0, z_1, z_2$: ten values (`primvars:athenea:splat:transferZonal`).
As harmonics a lobe about $\mathbf{a}$ is $g_{lm} = z_l\, k_l\, Y_{lm}(\mathbf{a})$ with $k_l = \sqrt{4\pi/(2l+1)}$.
For a fixed axis the best coefficients are the projection $z_l = k_l \sum_m f_{lm} Y_{lm}(\mathbf{a})$, leaving error
$\lVert f\rVert^2 - \sum_l z_l^2$, so the fit (an AOFX effect, `splattransferzonal`) searches the axis on the device
(band-1 direction, the normal, 64 Fibonacci directions, then a halving walk), fits the second lobe to the residual and
refits the pair twice (Sloan 2008). At draw time `splatTransferFrame` turns each axis by the skinned frame and rebuilds
nine world harmonics, which everything downstream reads unchanged. The bake poses the cloud at `--time` with the
engine's own skinner, so the frame the lobes are fitted against is the packed quaternion a renderer reads. The shadow
bits stay two words, re-binned into the Gaussian's frame. The indirect half is not kept zonal.

## 4.5 The body

For a Gaussian with a transfer, the part that does not depend on the eye is

$$\mathbf{B} = \sum_{k<16} L_k\,\big(T_k^{\text{dir}} + T_k^{\text{ind}}\big) \;+\; E_\odot\,\frac{\max(\mathbf{n}\cdot\mathbf{s},0)}{\pi}\, O(\mathbf{s}) \;+\; E_\odot\, T^{\text{ind}}(\mathbf{s}),
\qquad T^{\text{ind}}(\mathbf{s}) = \max\Big(\sum_k T_k^{\text{ind}} Y_k(\mathbf{s}), 0\Big),$$

the last term being the sun's bounce, which the first transfer lost with the sun it took out of the harmonics. $O(\mathbf{s})$
is the sun's share: for TX the bilinear read of the four cells around $\mathbf{s}$, cells behind the normal left out;
for the 64 bits the same on an 8 x 8 grid; without bits, the degree-2 ratio

$$O_{\text{SH}}(\mathbf{s}) = \frac{\sum_k T_k Y_k(\mathbf{s})}{\sum_k \hat A_l\, Y_k(\mathbf{n})\, Y_k(\mathbf{s})}, \qquad \hat A_l = \{1, 2/3, 1/4\},$$

numerator and denominator sharing the truncation so it reads one where nothing occludes. The diffuse body is then
$U\,\rho\,(1-m)\,\big[(1-t)\,\mathbf{B} + t\,(\ldots)\big]$ with $U$ the layers' throughput (section 4.6.4) and $t$ the
transmission. $\mathbf{B}$, the sun's share and the field coupling (below) are kept per Gaussian while the sky and the
lights stand (`transferViewless`, the `txCache`); a frame computes only what the eye changes.

The transfer's constant term is also an ambient occlusion: over an open hemisphere $T_0 = A_0 Y_0 / \pi = 0.28209$, so
$o = \operatorname{sat}(T_0 / 0.28209)$ is the fraction of the sky that reaches the Gaussian.

**Lights that are not the sky.** A distant, sphere, disk or rect light is evaluated with one sample at its centre
through the same lobes. With TX cells, its shadow is the cells read over the cone the light subtends
($\tan^{-1}$ of its radius over the distance, or half a distant light's angular diameter), the darker of that and any
ray-measured shadow; its bounce is $T^{\text{ind}}$ read along it, times $L/p$ of the sample, on the body and the sheen,
and it couples the reflected field as the sun does (`splatLightBounce`).

## 4.6 Reflection

### 4.6.1 Which part of a lobe sees the sky

The first transfer narrowed every reflection by Lagarde and de Rousiers' fit of the constant term,

$$O_{\text{L}}(\mathbf{n}\cdot\boldsymbol{\omega}_o, o, r) = \operatorname{sat}\!\Big( (\mathbf{n}\cdot\boldsymbol{\omega}_o + o)^{2^{-16r-1}} - 1 + o \Big),$$

one number for the hemisphere: a door reflected the open sky where the ground stands in its mirror, and the Corvette's
rims read 2.4 times the path-traced mesh, `Metal_rough` 6.5 times, the brake discs a hundred. With TX cells the lobe
itself is asked (`splatLobeOpen`, `splatConeOpen`): the cell read $o_c(\cdot)$ (bilinear over four cells, cells behind
$\mathbf{n}$ excluded) at the mirror and at a ring of six directions around it at angle

$$\theta_r = \max\!\big(\min(1.4\,\tan^{-1}\alpha,\ 1.2),\ 2\cdot\tfrac{\pi}{2S_c}\big),$$

about the angle within which GGX reflects half its energy, never narrower than a cell; and

$$O(\mathbf{m}, \alpha) = \frac{\tfrac13 o_c(\mathbf{m}) + \tfrac19 \sum_{k=1}^{6} o_c(\mathbf{d}_k)}{\tfrac13 + \tfrac69}$$

over the reads that weighed anything. A cone inside one cell read that cell's single bit and speckled the Corvette's
paint (0.393 relMSE against the path trace; 0.298 with the cone held to a cell).

### 4.6.2 What the closed part shows: the field coupled to the sky

The field was baked under a white sky. Under a real one its pattern stands and its brightness is scaled per channel by
how much more the surroundings bounce under that sky than under white (`splatFieldCoupling`):

$$\kappa = \frac{\sum_k T_k^{\text{ind}} L_k + E_\odot\, T^{\text{ind}}(\mathbf{s})}{\sqrt{4\pi}\; T_0^{\text{ind}}},$$

falling back to the sky's mean radiance where the Gaussian bounces nothing. The coupling is rank one: exact for a
Gaussian surrounded by one surface lit by one sky, and it puts a lit ground's colour, not its shape, under a sky lit from
one side. Read along a direction at a lobe's roughness, the field's bands are narrowed as a reflected GGX taken as a
von Mises-Fisher of concentration $1/2\alpha^2$:

$$F(\mathbf{d}; \alpha) = \max\Big(\sum_{j<16} F_j\, e^{-l_j(l_j+1)\alpha^2}\, Y_j(\mathbf{d}),\ 0\Big).$$

The two roughnesses a Gaussian needs (polish and coat) are taken from one read of the 48 values
(`splatFieldAlongPair`). The alternatives weighed were a 9 x 9 transfer matrix per Gaussian (243 numbers, 5 GB for the
car, still degree 2) and a per-direction index into the cloud (needs a radiance cache and sees nothing of a mesh
ground).

### 4.6.3 Polish and coat

The base's reflection ("polish") and the coat are

$$\mathbf{R}_{\text{polish}} = U\,g\,P(\mathbf{n}\cdot\boldsymbol{\omega}_o)\,\big[O\,\tilde L(\mathbf{m}, r) + (1-O)\,\chi\,\kappa\,F(\mathbf{m};\alpha)\big] + U\,g\,P\,E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m}, \mathbf{s}),$$
$$\mathbf{R}_{\text{coat}} = w_c E_c\,\big[O_c\,\tilde L(\mathbf{m}, r_c) + (1-O_c)\,\kappa\,F(\mathbf{m};\alpha_c)\big] + w_c E_c\, E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m},\mathbf{s}),$$

where $P$ is the polish's environment albedo, $g$ the thin-wall gain (section 4.7), $E_c$ the coat's albedo and $\chi$
the polish-field gate:

- **The polish's openness and gate.** $\chi = 1$ and $O = O(\mathbf{m}, \alpha)$ only where the material's own
  roughness is at least `kPolishFieldRoughness` = 0.3; elsewhere $\chi = 0$ and $O = O_{\text{L}}$. On the Corvette's
  chrome (0) and rough metal (0.22) the cells' openness read 1.64 and 0.221 relMSE against the path trace where the
  harmonics' read 1.05 and 0.128 for the same mean, and the field read the metals a third bright (chrome 0.261 where the
  path trace reads 0.197). A lobe at 0.3 or wider is wider than the field's degree resolves, so what closes it (a ball's
  ground) is what the field says: green and white balls at 0.3421 on a grey ground reflected the sky in their lower half
  without it. The gate reads the *material* roughness, not the stored one, because the bake's widening pushes a thin
  chrome tube's stored roughness past 0.3. The coat always uses the cells and the field.
- **The analytic sun.** The map holds no sun, so the sun is added through each lobe's distribution of reflected
  directions with the eye along the normal, $D(\mathbf{h})/4$ for the half vector between mirror and sun, the disc's half
  angle $\theta_\odot$ folded into the width:

  $$p_\odot(\mathbf{m}, \mathbf{s}) = \frac{\tilde\alpha^2}{4\pi\,d^2},\qquad d = \cos^2\!\tfrac{\theta}{2}\,(\tilde\alpha^2-1)+1,\qquad \tilde\alpha^2 = \alpha^2 + \theta_\odot^2,\qquad \cos\theta = \mathbf{m}\cdot\mathbf{s}.$$

  It integrates to one, so weight times $E_\odot$ times $p_\odot$ is the sun's whole reflection. The split sum read a
  compact sun a quarter short at every map level (0.57-0.74 of the path trace on a floating metal ball; part of that gap
  came from the path tracer's own dome sampling density, since corrected).

### 4.6.4 Layers, Fresnel, metals, sheen

**Environment albedo.** $P$ is MaterialX's `ggxDirectionalAlbedo` with multiple-scattering compensation, not Lazarov's
fit, which differed by up to 76 % at grazing; a dielectric uses the exact Fresnel against $F_0$, a conductor its colour
at both ends, mixed by $m$. Where the material's metal is a generalized Schlick (OpenPBR, glTF), it runs from its colour
head on to the specular colour at grazing (`ggxEnvSchlickMetal`); for a dark metal a conductor of artistic index and a
Schlick differ by twice at sixty degrees, and the car's paint read matte and bright as a conductor.

**The coat's Fresnel.** Schlick is low through the middle of a dielectric's curve (0.070 at sixty degrees where a
lacquer at 1.45 reflects 0.081); a coat over black read 0.084 against the path trace's 0.099. The coat's albedo is the
fit scaled by exact over Schlick at the eye, fading with roughness:

$$E_c = E_{\text{fit}}(r_c, \mathbf{n}\cdot\boldsymbol{\omega}_o, \eta_c)\cdot \operatorname{lerp}\!\Big(\frac{F(\mathbf{n}\cdot\boldsymbol{\omega}_o,\eta_c)}{F_{\text{Schlick}}},\ 1,\ \operatorname{sat}(\alpha_c)\Big),$$

and exactly $F$ below $\alpha_c = 10^{-3}$. Against a quadrature of the lobe it is within 3 % at $\alpha$ 0.3 and 0.6
where the fit was 8-16 % under.

**Layering.** As MaterialX's `layer` node: the coat over the sheen over the base, each layer taking from what is below
it the share it reflects at the eye. With $E_{\text{fit}}$ the coat's fitted albedo, $E_s$ the sheen's directional albedo and $w_s$
the sheen colour's largest channel,

$$U = \big(1 - w_c E_{\text{fit}}\big)\big(1 - w_s E_s\big)\cdot \operatorname{lerp}\!\Big(1,\ \frac{1-K}{1-E_b K},\ w_c\Big),\qquad K = 1 - \frac{1-F_0(\eta_c)}{\eta_c^2},$$

the last factor OpenPBR's coat darkening (light returning from the base is reflected back down by the coat's
underside), with $E_b$ the base albedo; the paint's ball read 1.5 times the mesh at its centre without it. The sheen
(Imageworks, Conty and Kulla) is a lobe as broad as a body, so it is answered as one: its directional albedo times
$\mathbf{B}$, under the coat.

## 4.7 Glass, transmission and thin walls

A Gaussian cannot refract. What it sends on is the light from behind it: $(1-t)$ of the body faces the eye's side,
$t$ faces away, and what stands behind is seen through the Gaussian's opacity.

- **What passes is what is not reflected**: the transmitted half is weighted by $1 - w_{\text{spec}} E_{\text{fit}}(r,\eta)$, with $w_{\text{spec}}$ the specular weight; one
  Fresnel per face, as the path tracer's transmission lobe pays it; a ball came back half again too bright with a
  rim 2.5 times too hot before.
- **Rasterised TX glass** sees through along $-\boldsymbol{\omega}_o$: the sky where the cells behind are open
  (`splatConeOpen` about $-\mathbf{n}$) and $\kappa F$ where they are closed, so a windscreen shows the cabin, blurred
  to degree 3. **A solid glass is a lens**: its own far face closes every cell behind it, so the field's white-sky value
  along $-\boldsymbol{\omega}_o$ decides openness ($\operatorname{smoothstep}(0.5, 0.85)$ of its luminance) and the open
  part reads the sky sharp along the exit of a sphere guessed from the stored mean curvature (`lensExit`: bent in, across
  the chord, bent out), with the far face's Fresnel and second tint (`lensExitThrough`). Filling the lens with the field
  drew the pawn's clear head milky and flat (0.58 against the path-traced mesh).
- **Roughness through glass**: a microfacet tilt $\theta$ turns a mirror ray by $2\theta$ but a transmitted one by far
  less, so the sky through both faces is read at $r_t = r\sqrt{\operatorname{sat}((\eta - 1)/\sqrt2)}$; the pawn's head
  went from 0.081 to 0.048 relMSE.
- **The far face's reflection** sends back $(1-R)\,R\,(1-R)$ times the tint of the light along the mirror (the Gaussian's colour carries the first tint); a clear
  ball under a uniform sky read 0.952 without it, 0.9998 with it (0.92 through both faces + 0.04 near polish + 0.037
  this).
- **Thin walls.** A slab classified thin (open edges, or a thickness $2V/A$ under four cells or a fiftieth of its own size) is converted with
  opacity $2R_0/(1+R_0)$, $R_0 = ((\eta-1)/(\eta+1))^2$, and its reflection scaled by the gain $g = 1/R_0$ at its own
  index ($R_0$ held at $10^{-3}$ at least; 1.5 only where no index was kept), so the blend leaves the reflection whole. The
  Corvette's headlight cover at 1.15 got 0.0097 of opacity and, at the old 1.5 floor, a gain of 25 where 205 makes it
  whole. A sheet's Gaussians are drawn at an opacity of at least 0.1 with their colour scaled down by as much (and its
  ramp with it): windshield relMSE 2.26 to 0.89 under the shop, 0.31 to 0.067 under goegap.

## 4.8 Reflection across the footprint

The paint's error is one of sharpness, not energy: against the path trace blurred by one pixel its relMSE falls from
0.80 to 0.28. A Gaussian showed its centre's reflection flat over its footprint.

### 4.8.1 The per-pixel normal turn

The conversion writes the shape operator of the centre triangle in the Gaussian's axes $(\mathbf{u}, \mathbf{v})$,
$S = \operatorname{sym}\big([\Delta\mathbf{n}_1\ \Delta\mathbf{n}_2]\,[\Delta\mathbf{p}_1\ \Delta\mathbf{p}_2]^{-1}\big)$ projected on
$(\mathbf{u},\mathbf{v})$ (`m2sShape`). The projection inverts its own linearisation on the splat's plane: with
$J$ the $2\times2$ map from plane coordinates to pixels, one pixel right and down are $\mathbf{e}_x = J^{-1}(1,0)$,
$\mathbf{e}_y = J^{-1}(0,1)$, and the normal a pixel away is

$$\mathbf{n}_x = \operatorname{normalize}\big(\mathbf{n} + \mathbf{h}_x\big),\quad \mathbf{h}_x = \operatorname{clamp}_{0.5}\!\big((S\,\mathbf{e}_x)_u\,\mathbf{u} + (S\,\mathbf{e}_x)_v\,\mathbf{v}\big),$$

the turn held to half a unit so a splat seen edge on does not swing across the sky.

### 4.8.2 The slope, and the sharp lobe

For a rough reflection the projection reads the sky again along $\operatorname{reflect}(-\boldsymbol{\omega}_o, \mathbf{n}_x)$
and $\mathbf{n}_y$ and stores the colour's change per pixel, $\mathbf{g}_x, \mathbf{g}_y$ (`kSlopeMark`). The blend
draws $\max(\mathbf{c} + \mathbf{g}_x d_x + \mathbf{g}_y d_y, 0)$ at offset $\mathbf{d}$. The ramp's largest fall over
the three-sigma ellipse is held to the colour per channel, $3\sqrt{\mathbf{g}^\top\Sigma\,\mathbf{g}} \le \mathbf{c}$, so
the shade never reaches zero inside the footprint (a dashed black line along a pawn's gold ring before). The one-sided
slope beat the central difference and minmod on the Corvette (paint 0.393 against 0.584 and 0.564 relMSE).

A linear ramp cannot draw a point. A coat smoother than `kSharpRoughness` (0.2) and a base polish smoother than
`kSharpPolishRoughness` (0.5) are instead handed to the blend (`kSharpMark`): the projection subtracts the centre's
reading (sky and analytic sun) from the colour, unclamped, and writes the mirror's octahedral coordinate
$\mathbf{a}_0$ and its steps $\Delta\mathbf{a}_x, \Delta\mathbf{a}_y$ (zeroed across the octahedron's fold), the lobes'
weights (coat weight times $O_c$; polish rgb weight times $O$), their roughness and their sun weights. Per pixel the
blend decodes $\mathbf{m}(\mathbf{d}) = \operatorname{octDecode}(\mathbf{a}_0 + \Delta\mathbf{a}_x d_x + \Delta\mathbf{a}_y d_y)$
and adds

$$w_c\,\tilde L(\mathbf{m}(\mathbf{d}), r_c') + \mathbf{w}_p\,\tilde L(\mathbf{m}(\mathbf{d}), r_p') + E_\odot\big(\mathbf{w}_{p\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_p') + w_{c\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_c')\big),$$

clamping after. A 1.5-degree sun mirrored in a curved coat is drawn where it falls inside the Gaussian rather than as
a blob of the Gaussian's size. Cost: one sky read per pixel per sharp record, and the slope's group memory at 17,412 of
WebGPU's 17,416 bytes.

### 4.8.3 The pixel's own spread

The sharp read is a point sample at the pixel's centre, while the normal turns by $\mathbf{h}_x, \mathbf{h}_y$ across
the pixel: a box of slopes whose variance, doubled for the mirror, widens the lobe:

$$\alpha'^2 = \alpha^2 + 2\cdot\frac{\lVert\mathbf{h}_x\rVert^2 + \lVert\mathbf{h}_y\rVert^2}{12},$$

held at $\alpha' \le 1$, applied after the unwidening and the fold widening of section 4.9. A
panel's rolled lip no longer sparkles. In the WebGPU player's sharp lobe (§7), without this term, the door's and the sill's folds show
bright specks: with it, 27/25 specks become 2/1 (on the 1280 px car the player reads the door at 35.53 dB and the sill at
32.36 against the path trace, against 33.77 and 31.87 without the term and with the 640 px bake).

## 4.9 Undoing the bake's widening

### 4.9.1 Toksvig at conversion

A Gaussian has one normal. Where a thin curved strip of chrome turns the reflection across a Gaussian, the path-traced
pixel averages a moving highlight into a thin line, and a Gaussian drew a bead of its own size at full brightness (the
Corvette's chrome read 1.6 times the path trace). The conversion widens the roughness, and the coat's, by the spread of
normals under the Gaussian (Toksvig):

$$s = \lambda\, \max_{\text{edges } kj} \frac{\angle(\mathbf{n}_k, \mathbf{n}_j)}{\lVert\mathbf{p}_j - \mathbf{p}_k\rVert}\, \max(\sigma_u, \sigma_v),\qquad \alpha'^2 = \alpha^2 + 2 s^2,$$

with $\lambda$ = `--specular-filter` (1 with `--transfer`) and $r' = \operatorname{sat}(\alpha'^{1/2})$.

### 4.9.2 Unwidened sharp lobes

The sharp lobes already turn the mirror per pixel; read at the widened roughness, the turn is counted twice. A door's
lacquer, a mirror ($r_c = 0$), came out matte: stored at $p_{50} = 0.112$ on the door and 0.080 on the hood (640 px
bake), saturating to 1 at folds. The projection recovers the spread from the stored curvature,

$$s \approx \kappa_{\max}\, \max(\sigma_u, \sigma_v),\qquad \kappa_{\max} = \Big|\frac{S_{uu}+S_{vv}}{2}\Big| + \sqrt{\Big(\frac{S_{uu}-S_{vv}}{2}\Big)^2 + S_{uv}^2},$$

and reads the sharp coat and polish at

$$r = \big(r'^4 - 2 s^2\big)^{1/4}\quad (\text{held at } 0).$$

### 4.9.3 The fold's coat

At a fold the widened lacquer reached nearly 1 and a joint drew a dark streak (door joints 0.69 of the path trace). The
coat is held:

$$r_c \leftarrow \max\big(\operatorname{unwidened}(r_c, s),\ \min(r_c,\ 0.3)\big)\quad\text{if } r_{\text{mat}} \ge 0.3,$$

`kCoatFoldRoughness` = 0.3. Applied everywhere, it lit chrome, glass and a rim's spokes the path trace shows dark
(chrome -1.4 dB, spokes -0.5 dB by region in the web player), hence the paint-only gate. At 1280 px the bake no longer
widens the coat (it comes out at exactly 0), so on the re-baked car this term changes nothing.

### 4.9.4 A tube turns, a fold widens

The per-pixel turn extrapolates the centre triangle's shape operator; past an edge it no longer holds. The stored normal
and curvature are exact at the centre (normal $p_{90}$ 0.009 degrees from the interpolated mesh normal), but 64 % of a
hood's $\pm2\sigma$ footprints cross an edge, and the extrapolated normal's RMS error reached $p_{90}$ 46 and $p_{99}$ 198
degrees. Let $t = 2s$ be the turn across $\pm2\sigma$. In the 1280 px cloud a chrome tube's body sits at $t \in [0.3, 1]$,
coherent with its neighbours (2 % of opposite sign); normals smoothed across a fold within triangles narrower than the
Gaussian sit past 1 (a sill's 5-9 rad). With

$$w = \operatorname{smoothstep}(0.7,\ 1.5,\ t),\qquad S \leftarrow (1-w)\,S,\qquad \alpha' = \min\!\Big(\sqrt{\alpha^2 + w\,(t/2)^2},\ 0.5^2\Big)$$

(`kFoldTurnLow` 0.7, `kFoldTurnHigh` 1.5, `foldWidened` capped at `kFoldRoughness` 0.5 in $r$, never narrower than
before), a tube turns its pixels in full and draws its thin line of light, while a fold is integrated as the path tracer
integrates it within a pixel. The smoothstep keeps a moving camera from seeing a step.

## 4.10 Measurements against the path trace

All car figures are against the mesh path trace (`/World/Camera`, 1920 x 1080, `autoshop_01_4k`). "Car crop" is PSNR
on the 900 x 420 crop of the car, "frame" on the whole frame; class ratios are a region's mean over the path trace's.
"Leaves" is the whole baked cloud drawn without a cut; "cut" is the error cut of §6.3 at 0.25 px; "cull" is the
back-face culling of §6.6. The *widened* configurations read the sharp lobes at the bake's stored roughness (§4.9.1),
the *unwidened* ones apply §4.9.2.

### 4.10.1 The TX gate, per material (relMSE)

| state | chrome | rough metal | paint |
|---|---|---|---|
| first transfer | 0.536 | 0.0896 | 0.705 |
| TX, cells and field everywhere | 1.64 | 0.221 | 0.393 |
| polish open by harmonics, cone ≥ cell, field on coat only | 0.591 | 0.0835 | 0.341 |
| later: + sharp coat and sharp polish per pixel | 0.545 | 0.083 | 0.281 |

### 4.10.2 Playback cost

The whole TX Corvette (14.7 M Gaussians, no ground, 1920 x 1080) drew in 158 ms against about 83 for a relit cloud of
its size; of the projection's 84.5 ms the field and its coupling took 24, the cells read by the lobes 16.5, the sun's
share and bounce 12, the body's transfer 7.5. Reading the field once for both lobes (§4.6.2) and caching the
eye-independent terms (§4.5) address the first and the last.

### 4.10.3 The whole car, by bake and configuration

| bake | configuration | car crop (dB) | frame (dB) |
|---|---|---|---|
| 640 px | leaves (5.38 M), widened | 16.37 | 21.92 |
| 640 px | leaves, unwidened sharp lobes | 16.58 | 22.15 |
| 640 px | cut + cull, widened | 16.29 | -- |
| 640 px | cut, unwidened + coat held at folds (*cut A*) | 16.51 | 22.15 |
| 640 px | *cut A* + polish field on rough materials (*cut B*) | 16.55 | 22.18 |
| 640 px | leaves (cut at 0.01 px) with all of *cut B*'s terms, no cull / cull | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | *cut B* + turn capped at 0.2 rad / + coat fold on paint only | 16.57 / 16.57 | 22.20 / 22.20 |
| 1280 px | **cut + cull, capped turn, coat fold on paint only (5.0 M)** | **17.88** | **23.90** |
| 1280 px | the same without the cap | 17.88 | 23.91 |
| 1280 px | capped + sharp polish gated on unwidened roughness | 17.88 | 23.90 |
| 1280 px | capped + gate + tube/fold split | 17.90 | 23.93 |
| 1280 px | capped + tube/fold split (retained) | 17.88 | 23.91 |

PSNR over a crop moves little; the class ratios below show where the changes act.

### 4.10.4 The door and the hood, by class

The paint is a dark metal base (roughness 0.3421) under a coat of roughness 0 at 1.45. Each pixel is classed by where
its mirror falls: the ground (closed; the field answers) or the HDRI (open). With the 640 px leaves and widened lobes,
86 % of the door's deficit was in pixels reflecting ground. Mean over the path trace (640 px bake):

| door class | leaves, widened | cut + cull, widened | *cut A* | *cut B* | leaves, no cull | leaves, cull |
|---|---|---|---|---|---|---|
| mirror to ground < 3 m | 0.78 | 0.89 | 0.92 | **1.02** | 0.90 | 1.02 |
| mirror to ground 3-30 m | 0.69 | 0.77 | 0.76 | 0.88 | 0.79 | 0.88 |
| mirror to HDRI (open) | 0.92 | 1.04 | 1.03 | 1.13 | 1.01 | 1.13 |
| joints | 0.69 | 0.83 | 0.95 | **1.04** | 0.90 | 1.02 |

Unwidening alone leaves the ground classes at 0.78/0.67 and the joints at 0.69; the polish field (*cut B*) lifts the
near ground to 1.02 and the coat fold the joints to 1.04, as an analytic estimate of the missing term predicts
(under-coat 0.94 x polish albedo 0.078 x reflected ground 0.11 = 0.008 against a measured deficit of 0.0056-0.0085).
Back culling lifts the whole door by about 12 %. The open class's remaining 1.13 splits by whether the mirror meets the
car itself: 68 % of the excess is in "car" pixels (2.04 of the path trace), where the 16 x 16 cells of about 11 degrees
leave 25 % of sky open behind a mirror or pillar.

Local contrast (std of the high-pass over the mean, $\sigma$ = 1.5 / 4 px), relative to the path trace:

| | 640 px leaves, widened | 640 px leaves, unwidened | 640 px *cut B* | 1280 px cut + cull |
|---|---|---|---|---|
| door, open reflections | 0.12 / 0.26 | 0.40 / 0.61 | 0.35 / 0.53 | 0.56 / 0.74 |
| hood, open reflections | 0.27 / 0.38 | 0.41 / 0.52 | 0.41 / 0.51 | -- |

A simulation of the coat's reflection with the path trace's mirror over the prefiltered HDRI shows that the stored coat
roughness alone explains the blur: at $r_c$ = 0.11 the door keeps 0.24 / 0.31 of the $r_c$ = 0 contrast, at 0.08 the
hood 0.21 / 0.28.

### 4.10.5 The hood's highlights

Tiles of 24 px with highlights (168), aligned by high-pass correlation within $\pm$6 px:

| render | offset $p_{50}$ | $\sigma$ to match | peak (PT 5.8) |
|---|---|---|---|
| 640 px leaves, widened | 4.1 px | 2.5 | 1.9 |
| 640 px, no per-pixel turn | 4.1 px | 1.5 | 2.1 |
| 640 px, no slope | 4.2 px | 2.5 | 1.8 |
| 640 px leaves, unwidened | 3.2 px | 1.5 | 3.0 |
| 640 px *cut B* | 3.2 px | 1.5 | 2.6 |
| 1280 px cut + cull | 2.0 px | 0.75 | 3.3 |

The hood's mean goes from 0.89 to 0.98 of the path trace at 1280 px. The normal's extrapolation error scales with the
square of the cell (0.146 degrees median at 640 px, 0.060 at 960, 0.032 at 1280).

### 4.10.6 The extrapolated normal, and the tube/fold split

RMS error of the extrapolated normal on the hood (degrees):

| variant | $p_{50}$ | $p_{90}$ | $p_{99}$ |
|---|---|---|---|
| no turn | 0.575 | 22.7 | 35.8 |
| centre's shape operator | 0.146 | 46 | 198 |
| + cap $\lVert S\rVert\,2\sigma \le 0.2$ rad | 0.147 | 20.2 | 32.0 |
| least squares on mesh samples | 0.088 | 9.8 | 14.6 |

At 640 px the cap moves the car crop from 16.55 to 16.57 dB but worsens the front rim's joints (1.48 to 1.54); at
1280 px removing it changes nothing (§4.10.3). The tube/fold split of §4.9.4 replaces it. On the 1280 px cut:

| | capped turn | + sharp polish gate | + gate + split | + split (retained) |
|---|---|---|---|---|
| bright specks on the sill's fold | 10 | 11 | 1 | 1 |
| chrome strip, RMS | 0.0358 | 0.0535 | 0.0415 | 0.0359 |
| chrome strip, high-pass / PT | 0.41 | 1.12 | 0.58 | 0.41 |

The retained split also takes the door's specks from 405 to 389 and the rim's from 204 to 190, leaving frame and crop
unchanged. At 1280 px the bake no longer widens the coat, and without the split the Gaussians on the sill's fold reflect
as mirrors with spinning normals, bright dashes over the path trace's dark groove; the split removes them without
touching the strip. Alternatives considered: gating the sharp polish on the unwidened roughness triples the strip's
high-pass error; `--specular-filter 0` draws a dashed dark line on the sill's fold and sparkling chrome. The polish
needs its widening there.

## 4.11 Limitations

- **Rank-one field.** The coupling puts a lit ground's colour, not its shape, in a reflection under a sky lit from one
  side; at degree 3 the field resolves a horizon to about twenty degrees and blurs the car's shadow on the ground into
  the far ground (in simulation: correlation 0.56 at degree 3, 0.63 at degree 5, 0.85 with a 32 x 32 octahedral map).
- **Cell resolution.** 16 x 16 cells of about 11 degrees let sky through where a sharp coat mirrors the car itself
  (the door's open class at 1.13); 32 x 32 cells (+96 B per Gaussian) would halve it.
- **One sun per dome**; lamps inside a scene are shadowed by bits that say which ways leave the scene, not which reach
  the lamp. The bake has no lamp emission: the headlights are dark against the path trace.
- **Remaining defects** on the 1280 px car: the chrome strip on the front bumper 38 % brighter than the path trace with
  a lower peak; the rim's near-ground reflection at 0.72; the hood highlights 2.0 px off, which a curvature fitted over
  the footprint from the mesh would roughly halve.
- **Zonal transfers** keep the direct half only, and know only the pose that was traced; occlusion between limbs in
  other poses is not multiplied in.
