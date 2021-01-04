using DataFrames
using CSV
using JSONTables
using JSON3

function reformat(file; complete=true, csv_file="", json_file = "", to_csv=true, to_json=true)
    if to_csv
        if complete
            csvf = CSV.File(file, header=false)
            @info csvf
            l = length(csvf[1])
            column_types = vcat([Bool, String], fill(Int, l - 2), [String])

            df = DataFrame(CSV.File(file, types=column_types, drop=[2,l], header=false))

            column_names = map(i -> Symbol("x$i"), 1:(l - 3))
            rename!(df, vcat([:concept], column_names))

            new_file = isempty(csv_file) ? split(file, ".")[1] * ".csv" : csv_file

            CSV.write(new_file, df)
        else
            csvf = CSV.File(file, datarow=2, header=false)        
            @info csvf
            l = length(csvf[1])
            column_types = vcat(fill(Int, l - 1), [String])

            df = DataFrame(CSV.File(file, datarow=2, types=column_types, drop=[l], header=false))

            column_names = map(i -> Symbol("x$i"), 1:(l - 1))
            rename!(df, column_names)

            nb_sols = length(df.x1) ÷ 2
            df.concept = vcat(trues(nb_sols), falses(nb_sols))
            select!(df, vcat([:concept], column_names))

            new_file = isempty(csv_file) ? split(file, ".")[1] * ".csv" : csv_file

            CSV.write(new_file, df)

        end
    end
end

function reformat_folder(path; csv_path="", json_path="", to_csv=true, to_json=true)
    for f in readdir(path)
        csv_f = split(f, ".")[1] * ".csv"
        if occursin(r"data", f) || occursin(r"repart", f)
            !isempty(csv_path) && cp(joinpath(path, f), joinpath(csv_path, f))
        elseif occursin(r"complete", f)
            reformat(joinpath(path, f), csv_file=joinpath(csv_path, csv_f))
        else
            reformat(joinpath(path, f), complete=false, csv_file=joinpath(csv_path, csv_f))
        end
    end
end

# reformat("data/old/ad-6-6-100-la.txt", complete = false, csv_file = "data/ad-6-6-100-la.csv")
# reformat("data/old/complete_ad-4-4.txt", csv_file = "data/complete_ad-4-4.csv")
# reformat("data/old/complete_le-7-8-35.txt", csv_file = "data/complete_le-7-8-35.csv")

reformat_folder(joinpath(pwd(), "data", "old"); csv_path=joinpath(pwd(), "data", "csv"))