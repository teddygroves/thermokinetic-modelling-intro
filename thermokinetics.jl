"Functions for calculating fluxes and species dynamics in thermokinetic models."
module Thermokinetics

export OdeUnknowns,
       OdeInfo,
       get_reversibility,
       get_free_enzyme_ratio,
       get_allostery,
       get_saturation,
       get_reaction_flux,
       get_flux,
       Sv

using InvertedIndices
using LinearAlgebra
import Base.@kwdef

@kwdef struct OdeUnknowns
  enzyme::Vector{Float64}
  conc_unbalanced::Vector{Float64}
  kcat::Vector{Float64}
  dgf::Vector{Float64}
  km::Vector{Float64}
  ki::Vector{Float64}
  dc::Vector{Float64}
  tc::Vector{Float64}
end


@kwdef struct OdeInfo
  S::Matrix
  ix_balanced::Vector{Integer}
  subunits::Vector{Integer}
  sp_to_km::Vector{Dict{Integer,Integer}}
  sp_to_ki::Vector{Dict{Integer,Integer}}
  sp_to_dc::Vector{Dict{Integer,Integer}}
  allosteric_inhibitors::Vector{Vector{Integer}}
  allosteric_activators::Vector{Vector{Integer}}
  unknowns::OdeUnknowns
end

"Main function, in the format required by the DifferentialEquations library."
function Sv(conc_balanced::Vector, ode_info::OdeInfo, t::Float64)
  return ode_info.S[ode_info.ix_balanced,:] * 
    get_flux(conc_balanced, ode_info)
end


"Get flux for reactions in a network given current concentraiton and ode info."
function get_flux(conc_balanced::Vector, ode_info::OdeInfo)
  dgr_std = ode_info.S' * ode_info.unknowns.dgf
  conc = zeros(size(ode_info.S)[1])
  conc[ode_info.ix_balanced] = conc_balanced
  conc[InvertedIndex(ode_info.ix_balanced)] = ode_info.unknowns.conc_unbalanced
  print(conc)
  return [
    get_reaction_flux(
      conc,
      ode_info.S[:,r],
      ode_info.subunits[r],
      ode_info.sp_to_km[r],
      ode_info.sp_to_ki[r],
      ode_info.sp_to_dc[r],
      ode_info.allosteric_inhibitors[r],
      ode_info.allosteric_activators[r],
      ode_info.unknowns.enzyme[r],
      ode_info.unknowns.kcat[r],
      ode_info.unknowns.tc[r],
      dgr_std[r],
      ode_info.unknowns.km,
      ode_info.unknowns.ki,
      ode_info.unknowns.dc,
    )
    for r in 1:size(ode_info.S)[2]
  ]
end


"Get the flux through a reaction"
function get_reaction_flux(
  s::Vector{Float64},
  S_rxn::Vector{Float64},
  subunits::Integer,
  sp_to_km::Dict{Integer,Integer},
  sp_to_ki::Dict{Integer,Integer},
  sp_to_dc::Dict{Integer,Integer},
  ais::Array{Integer},
  aas::Array{Integer},
  enzyme::Float64,
  kcat::Float64,
  dgr_std::Float64,
  tc::Float64,
  km::Vector{Float64},
  ki::Vector{Float64},
  dc::Vector{Float64},
)
  return (
    enzyme
    * kcat
    * get_reversibility(dgr_std, S_rxn, s)
    * get_saturation(s, km, ki, S_rxn, sp_to_km, sp_to_ki)
    * get_allostery( 
      s, km, ki, tc, dc, S_rxn, sp_to_km, sp_to_ki, sp_to_dc, ais, aas, subunits
    )
  )
end


"Find the reversibility factor for a reaction."
function get_reversibility(
  dgr_std::Float64,
  S_rxn::Vector{Float64},
  c::Vector{Float64}
)
  RT = -0.008314 * 298.15
  return 1 - exp((dgr_std + RT * dot(S_rxn, log.(c)))/RT)
end


"Find the free enzyme ratio for a reversible reaction."
function get_free_enzyme_ratio(
  s::Vector{Float64},
  km::Vector{Float64},
  ki::Vector{Float64},
  S_rxn::Vector{Float64},
  sp_to_km::Dict{Integer,Integer},
  sp_to_ki::Dict{Integer,Integer}
)
  return - 1. +
  prod(1. + s[sp] / km[km_ix] for (sp, km_ix) in sp_to_km if S_rxn[sp] < 0) +
  prod(1. + s[sp] / km[km_ix] for (sp, km_ix) in sp_to_km if S_rxn[sp] > 0) +
  sum(Float64[s[sp]/ki[ki_ix] for (sp, ki_ix) in sp_to_ki])
end


"Find the saturation factor for a reaction."
function get_saturation(
  s::Vector{Float64},
  km::Vector{Float64},
  ki::Vector{Float64},
  S_rxn::Vector{Float64},
  sp_to_km::Dict{Integer,Integer},
  sp_to_ki::Dict{Integer,Integer}
)
  return prod(s[sp] / km[km_ix] for (sp, km_ix) in sp_to_km if S_rxn[sp] > 0) *
    get_free_enzyme_ratio(s, km, ki, S_rxn, sp_to_km, sp_to_ki)
end


"Find the allostery factor for a reaction."
function get_allostery(
  s::Vector{Float64},
  km::Vector{Float64},
  ki::Vector{Float64},
  tc::Float64,
  dc::Vector{Float64},
  S_rxn::Vector{Float64},
  sp_to_km::Dict{Integer,Integer},
  sp_to_ki::Dict{Integer,Integer},
  sp_to_dc::Dict{Integer,Integer},
  inhibitors::Vector{Integer},
  activators::Vector{Integer},
  subunits::Integer
)
  fer = get_free_enzyme_ratio(s, km, ki, S_rxn, sp_to_km, sp_to_ki)
  Qtense = 1 + sum(Float64[s[i]/dc[sp_to_dc[i]] for i in inhibitors])
  Qrelaxed = 1 + sum(Float64[s[a]/dc[sp_to_dc[a]] for a in activators])
  return inv(1 + tc * fer * (Qtense/Qrelaxed) ^ subunits)
end
end
