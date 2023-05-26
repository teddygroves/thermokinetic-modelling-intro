include("./thermokinetics.jl")
using .Thermokinetics
using DifferentialEquations

example_starting_concentration = [0.1, 0.4, 1.2, 0.1]
example_S = Matrix(
  [-1.  0.  0.;
    1. -1.  0.;
    0.  1. -1.;
    0.  0.  1.;]
)
example_unknowns = OdeUnknowns(
  enzyme=[0.5, 0.5, 0.5],
  kcat=[12., 1., 5.],
  dgf=[25., 25., 50., 50.],
  km=[1., 1., 1., 1., 1., 1.],
  tc=[0., 0., 1.2],                # NB all reactions must have a tc 
  ki=[1.5],
  dc=[12.5]
)
example_ode_info = OdeInfo(
  S=example_S,
  subunits=[1, 2, 1],
  sp_to_km=[Dict(1=>1, 2=>2), Dict(2=>3, 3=>4), Dict(3=>5, 4=>6)],
  sp_to_ki=[Dict([]), Dict(2=>1), Dict([])],
  sp_to_dc=[Dict([]), Dict([]), Dict(3=>1)],
  allosteric_inhibitors=[[], [], [3]],
  allosteric_activators=[[], [], []],
  unknowns=example_unknowns
)

example_Sv = Sv([1., 2., 3., 4.], example_ode_info, 0.1)
print(example_Sv)

tspan = (0., 20.)
prob = ODEProblem(Sv, example_starting_concentration, tspan, example_ode_info)
print(prob)
sol = solve(prob, TRBDF2(autodiff=false))
print(sol)
