using Random
using Printf
using JLD2

using Oceananigans
using Oceananigans.Units: minute, minutes, hour, hours

using Oceanostics.TKEBudgetTerms: BuoyancyProductionTerm, IsotropicKineticEnergyDissipationRate, TurbulentKineticEnergy, ZShearProductionRate

# ## The grid
#
grid = RectilinearGrid(GPU();
                       size = (256, 256, 256),
                     extent = (256, 256,  96))
# ### The Stokes Drift profile
#
using Oceananigans.BuoyancyModels: g_Earth

 amplitude = 1.423 # m
wavelength = 60  # m
wavenumber = 2π / wavelength # m⁻¹
 frequency = sqrt(g_Earth * wavenumber) # s⁻¹

const vertical_scale = wavelength / 4π
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹

# The Stokes drift profile is
uˢ(z) = Uˢ * exp(z / vertical_scale)

# and its `z`-derivative is
∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

# ## Buoyancy that depends on temperature and salinity
#
# We use the `SeawaterBuoyancy` model with a linear equation of state,

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=8e-4))

#
# ## Boundary conditions
#

Qᵀ = 1.221e-4 # K m s⁻¹, surface _temperature_ flux

# Finally, we impose a temperature gradient `dTdz` both initially and at the
# bottom of the domain, culminating in the boundary conditions on temperature,
dTdz = 0.01 # K m⁻¹
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# Note that a positive temperature flux at the surface of the ocean
# implies cooling. This is because a positive temperature flux implies
# that temperature is fluxed upwards, out of the ocean.
#

ustar = 0.0 # m s⁻¹, friction velocity
Qᵘ = - ustar*ustar # m² s⁻²
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵘ))

Qˢ = 0.0
S_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qˢ))

# initial condition
S₀ = 35.0
hb₀ = 33.0 # m
T₀ = 15.0
T0(z) = z < - hb₀ ? T₀ + dTdz * (z + hb₀) : T₀

# latitude
lat = 0.0

#####
##### Sponge layer
#####

gaussian_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

u_sponge = v_sponge = w_sponge = Relaxation(rate=1/hour, mask=gaussian_mask)

T_sponge = Relaxation(rate = 1/hour,
                      target = LinearTarget{:z}(intercept=T₀+dTdz*hb₀, gradient=dTdz),
                      mask = gaussian_mask)
S_sponge = Relaxation(rate = 1/hour,
                      target = S₀,
                      mask = gaussian_mask)

# ## Model instantiation
#
model = NonhydrostaticModel(
                advection = WENO(order=5),
              timestepper = :RungeKutta3,
                     grid = grid,
                  tracers = (:T, :S),
                 coriolis = FPlane(latitude=lat),
                 buoyancy = buoyancy,
                  closure = AnisotropicMinimumDissipation(),
             stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
      boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs),
                  forcing = (u=u_sponge, v=v_sponge, w=w_sponge, T=T_sponge, S=S_sponge))


# ## Initial conditions
#

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = T0(z) + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)

## `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=S₀)

# ## Setting up a simulation
#

simulation = Simulation(model, Δt=10.0, stop_time=48hours)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim),
                                prettytime(sim),
                                prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

# We then set up the simulation:

# ## Output
#
## Create a NamedTuple with eddy viscosity
eddy_viscosity = (; νₑ = model.diffusivity_fields.νₑ)

simulation.output_writers[:slices_xy] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xy.jld2",
                         indices = (:, :, grid.Nz-3),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_xy2] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xy2.jld2",
                         indices = (:, :, grid.Nz-7),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_xy3] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xy3.jld2",
                         indices = (:, :, grid.Nz-15),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_xy4] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xy4.jld2",
                         indices = (:, :, grid.Nz-39),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_xy5] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xy5.jld2",
                         indices = (:, :, grid.Nz-79),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_xz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_xz.jld2",
                         indices = (:, 1, :),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

simulation.output_writers[:slices_yz] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                         filename = "slices_yz.jld2",
                         indices = (1, :, :),
                         schedule = TimeInterval(30),
               overwrite_existing = true)

fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivity_fields.νₑ,))
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                     filename = "fields.jld2",
                     schedule = TimeInterval(3hours),
           overwrite_existing = true)

u, v, w = model.velocities
U = Field(Average(u, dims=(1, 2)))
V = Field(Average(v, dims=(1, 2)))
T = Field(Average(model.tracers.T, dims=(1, 2)))
wt = Average(w * model.tracers.T, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))
ww = Average(w * w, dims=(1, 2))
uu = Average((u-U) * (u-U), dims=(1, 2))
vv = Average((v-V) * (v-V), dims=(1, 2))
uv = Average((u-U) * (v-V), dims=(1, 2))
w3 = Average(w^3, dims=(1, 2))
tt = Average((model.tracers.T-T) * (model.tracers.T-T), dims=(1, 2))
wtsb = Average(-∂z(model.tracers.T) * model.diffusivity_fields.νₑ, dims=(1, 2))
wusb = Average(-∂z(u) * model.diffusivity_fields.νₑ, dims=(1, 2))
wvsb = Average(-∂z(v) * model.diffusivity_fields.νₑ, dims=(1, 2))
# TKE budget
tke = TurbulentKineticEnergy(model; U, V)
p = model.pressures.pHY′ == nothing ? model.pressures.pNHS : sum(model.pressures)
shear_production = ZShearProductionRate(model; U, V)
dissipation = IsotropicKineticEnergyDissipationRate(model; U, V)
buoyancy_flux = BuoyancyProductionTerm(model)
e = Average(tke, dims=(1, 2))
tke_shear_production = Average(shear_production, dims=(1, 2))
tke_dissipation = Average(dissipation, dims=(1, 2))
tke_advective_flux = Average(∂z(Field(w * tke)), dims=(1, 2))
tke_pressure_flux = Average(∂z(Field(w * p)), dims=(1, 2))
tke_buoyancy_flux = Average(buoyancy_flux, dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (u=U, v=V, T=T, wt=wt, wu=wu, wv=wv, ww=ww, vv=vv, uu=uu, uv=uv, w3=w3, tt=tt, wtsb=wtsb, wusb=wusb, wvsb=wvsb, e=e, tke_shear_production=tke_shear_production, tke_dissipation=tke_dissipation, tke_advective_flux=tke_advective_flux, tke_buoyancy_flux=tke_buoyancy_flux, tke_pressure_flux=tke_pressure_flux),
                     schedule = TimeInterval(30),
                     filename = "averages.jld2",
           overwrite_existing = true)

checkpointer = Checkpointer(model,
                            schedule = TimeInterval(12hours),
                  overwrite_existing = true)
simulation.output_writers[:checkpointer] = checkpointer

# We're ready:

run!(simulation)

