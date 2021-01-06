"""
    csv2space(file; filter=:none)
Convert a csv file into a collection of configurations. If the filter is set to `:concept`, only solutions will be extracted.
"""
function csv2space(file; filter=:none)
    df = DataFrame(CSV.File(file))
    filter == :solutions && (df = df[df.concept .== true,:])
    return Vector(map(Vector, eachrow(df[!, Not(:concept)])))
end
