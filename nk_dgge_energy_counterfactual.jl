###DSGE###

###Pakiety których używam###

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Plots

###Wczytanie danych do analizy kontrfaktycznej###

const DATA_FILE = "C:/Users/gerar/Desktop/IDUB-DSGE/Dane_IDUB_2012_2024_with_apostrophes.csv"

if isfile(DATA_FILE)
    println("Wczytuję dane z pliku: ", DATA_FILE)
    df = CSV.read(DATA_FILE, DataFrame; delim=';')

    println("Pierwsze wiersze danych:")
    println(first(df, 5))

    ###Czyszczenie, usuwa apostrofy, spacje, itp.
    function clean_rate(x)
        if x === missing
            return missing
        elseif x isa String
            s = strip(replace(x, "'" => ""))
            if isempty(s)
                return missing
            end
            v = tryparse(Float64, s)
            return v === nothing ? missing : v
        elseif x isa Number
            return Float64(x)
        else
            return missing
        end
    end

    df.interest_rate = clean_rate.(df.interest_rate)

    float_or_missing(x) = x === missing ? missing : Float64(x)

    df.inflation          = float_or_missing.(df.inflation)
    df.gdp_real           = float_or_missing.(df.gdp_real)
    df.energy_price_index = float_or_missing.(df.energy_price_index)

    println("\nPo czyszczeniu interest_rate:")
    println(first(df, 5))

    println("\nStatystyki opisowe (2012–2024):")

    function describe_series(name, v)
        v_clean = collect(skipmissing(v))
        println("---- ", name, " ----")
        if isempty(v_clean)
            println("Brak dostępnych danych (same missing).")
            return
        end
        println("średnia:     ", mean(v_clean))
        println("odch. stand: ", std(v_clean))
        if length(v_clean) > 1
            println("autokorelacja (lag 1): ",
                    cor(v_clean[1:end-1], v_clean[2:end]))
        end
    end

    describe_series("inflation", df.inflation)
    describe_series("interest_rate", df.interest_rate)
    describe_series("energy_price_index", df.energy_price_index)
    describe_series("gdp_real", df.gdp_real)
else
    @warn "Plik danych $(DATA_FILE) nie istnieje. Model działa teoretycznie, bez kalibracji do danych."
end

###Parametry modelu###

struct NKEnergyParams
    β::Float64     # czynnik dyskontowy
    σ::Float64     # elastyczność międzyokresowa substytucji (IS)
    κx::Float64    # wrażliwość inflacji na lukę popytową
    κe::Float64    # wpływ cen energii na inflację
    χ::Float64     # wpływ luki popytowej na ceny energii
    ϕπ::Float64    # reakcja stopy procentowej na inflację
    ϕx::Float64    # reakcja stopy na lukę popytową

    ρ_rN::Float64  # persystencja naturalnej stopy
    ρ_uπ::Float64  # persystencja szoku kosztowego
    ρ_aE::Float64  # persystencja szoku podaży energii
    ρ_εi::Float64  # persystencja szoku polityki pieniężnej
end

function default_params(; ϕπ::Float64 = 1.5)
    β     = 0.99
    σ     = 1.0
    κx    = 0.10
    κe    = 0.05
    χ     = 0.10
    ϕx    = 0.5

    ρ_rN  = 0.8
    ρ_uπ  = 0.5
    ρ_aE  = 0.7
    ρ_εi  = 0.5

    return NKEnergyParams(β, σ, κx, κe, χ, ϕπ, ϕx,
                          ρ_rN, ρ_uπ, ρ_aE, ρ_εi)
end

###########################################################
# 3. Macierz strukturalna A (równania w formie A*y_t = b_t)
###########################################################
# y_t = [x_t, π_t, pE_t, i_t]'
#
# 1) x_t - (1/σ)i_t = E_t x_{t+1} + (1/σ)(E_t π_{t+1} + rN_t)
# 2) -κx x_t + π_t - κe pE_t      = β E_t π_{t+1} + uπ_t
# 3) -χ x_t + pE_t                = aE_t
# 4) -ϕx x_t - ϕπ π_t + i_t      = εi_t
###########################################################

function build_A(p::NKEnergyParams)
    return [
        1.0        0.0        0.0        1.0/p.σ
       -p.κx       1.0       -p.κe       0.0
       -p.χ        0.0        1.0        0.0
       -p.ϕx      -p.ϕπ       0.0        1.0
    ]
end

###Symulacja IRF przy AR(1)###

"""
simulate_irf(p; T, shock_type, shock_size)

Zwraca NamedTuple:
(x, π, pE, i, rN, uπ, aE, εi)
"""
function simulate_irf(p::NKEnergyParams;
                      T::Int = 40,
                      shock_type::Symbol = :monetary,
                      shock_size::Float64 = 0.01)

    A = build_A(p)

    ###Zmienne endogeniczne.
    x   = zeros(Float64, T+1)
    π   = zeros(Float64, T+1)
    pE  = zeros(Float64, T)
    i   = zeros(Float64, T)

    ###Szoki / procesy egzogeniczne.
    rN  = zeros(Float64, T)   # naturalna realna stopa
    uπ  = zeros(Float64, T)   # szok kosztowy
    aE  = zeros(Float64, T)   # szok podaży energii
    εi  = zeros(Float64, T)   # czysty szok polityki pieniężnej

    ###Początkowy szok, zależnie od typu.
    if shock_type == :monetary
        εi[1] = shock_size
    elseif shock_type == :energy
        aE[1] = shock_size
    elseif shock_type == :natural_rate
        rN[1] = shock_size
    elseif shock_type == :cost_push
        uπ[1] = shock_size
    else
        error("Nieznany typ szoku: $shock_type")
    end

    ###AR(1) w szokach.
    for t in 2:T
        rN[t] = p.ρ_rN * rN[t-1]
        uπ[t] = p.ρ_uπ * uπ[t-1]
        aE[t] = p.ρ_aE * aE[t-1]
        εi[t] = p.ρ_εi * εi[t-1]
    end

    ###Warunki końcowe.
    x[T+1] = 0.0
    π[T+1] = 0.0

    # Rozwiązanie wstecz: t = T, ..., 1.
    for t in T:-1:1
        Ex_next = x[t+1]
        Eπ_next = π[t+1]

        b = [
            Ex_next + (1.0/p.σ)*(Eπ_next + rN[t])
            p.β * Eπ_next + uπ[t]
            aE[t]
            εi[t]
        ]

        y = A \ b

        x[t]  = y[1]
        π[t]  = y[2]
        pE[t] = y[3]
        i[t]  = y[4]
    end

    return (x=x[1:T], π=π[1:T], pE=pE, i=i,
            rN=rN, uπ=uπ, aE=aE, εi=εi)
end

###Analiza kontrfaktyczna w 2019, czyli bardziej restrykcyjna polityka###

"""
run_counterfactual_2019(T, shock_size)

Interpretacja:
t = 1 oznacza 2019Q1,
szok np. polityki pieniężnej albo szok energii.
"""
function run_counterfactual_2019(; T::Int = 16,
                                 shock_size::Float64 = 0.01,
                                 shock_type::Symbol = :monetary)

    ###Polityka „bazowa” (łagodniejsza)
    p_base   = default_params(ϕπ = 1.5)

    ###Polityka „bardziej restrykcyjna”
    p_strict = default_params(ϕπ = 2.0)

    irf_base   = simulate_irf(p_base;   T=T,
                              shock_type=shock_type,
                              shock_size=shock_size)
    irf_strict = simulate_irf(p_strict; T=T,
                              shock_type=shock_type,
                              shock_size=shock_size)

    println("\nKontrfaktyk: ϕπ = 2.0 vs 1.5")
    println("Interpretacja: t=1 → 2019Q1\n")

    println("Inflacja (pierwsze 8 kwartałów):")
    println("Bazowa:          ", irf_base.π[1:8])
    println("Bardziej restr.: ", irf_strict.π[1:8])

    println("\nCeny energii (pierwsze 8 kwartałów):")
    println("Bazowa:          ", irf_base.pE[1:8])
    println("Bardziej restr.: ", irf_strict.pE[1:8])

    return (base=irf_base, strict=irf_strict)
end

###Wykresy IRF###

function plot_irfs(result; Tshow::Int = 16)

    irf_base   = result.base
    irf_strict = result.strict

    T = min(Tshow, length(irf_base.π))
    t = 1:T

    default(
        fontfamily = "CMU Serif",
        legend = :topright,
        grid = true,
        gridalpha = 0.3,
        linewidth = 3,
        framestyle = :box,
        background_color = "#F2F2F2",
        guidefont = font(14),
        tickfont  = font(12),
        legendfont = font(12)
    )

    purple = "#6A5ACD"
    mint   = "#2ECC71"

    ###Inflacja
    p1 = plot(t, irf_base.π[1:T], color=purple, label="ϕπ = 1.5",
        title="IRF: Inflacja", xlabel="Kwartał", ylabel="π")
    plot!(p1, t, irf_strict.π[1:T], color=mint, linestyle=:dash,
        linewidth=2, label="ϕπ = 2.0")

    ###Luka popytowa.
    p2 = plot(t, irf_base.x[1:T], color=purple, label="ϕπ = 1.5",
        title="IRF: Luka popytowa", xlabel="Kwartał", ylabel="x")
    plot!(p2, t, irf_strict.x[1:T], color=mint, linestyle=:dash,
        linewidth=2, label="ϕπ = 2.0")

    ###Ceny energii.
    p3 = plot(t, irf_base.pE[1:T], color=purple, label="ϕπ = 1.5",
        title="IRF: Ceny energii", xlabel="Kwartał", ylabel="pᴱ")
    plot!(p3, t, irf_strict.pE[1:T], color=mint, linestyle=:dash,
        linewidth=2, label="ϕπ = 2.0")

    ####Stopa procentowa.
    p4 = plot(t, irf_base.i[1:T], color=purple, label="ϕπ = 1.5",
        title="IRF: Stopa procentowa", xlabel="Kwartał", ylabel="i")
    plot!(p4, t, irf_strict.i[1:T], color=mint, linestyle=:dash,
        linewidth=2, label="ϕπ = 2.0")

    plt = plot(p1, p2, p3, p4; layout=(4,1), size=(900,1200), dpi=140)
    return plt
end

###Odpalenie kontrfaktyki###

println("\n=== KONTRAFAKTYKA 2019 (ϕπ 1.5 vs 2.0) ===")
result_2019 = run_counterfactual_2019(T=16,
                                      shock_size=0.01,
                                      shock_type=:monetary)

println("\n=== IRFy (bazowa vs restrykcyjna polityka) ===")
plt = plot_irfs(result_2019; Tshow=16)
display(plt)
