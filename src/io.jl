function csv2space(file; filter=:none)
    df = DataFrame(CSV.File(file))
    filter == :solutions && (df = df[df.concept .== true,:])
    return Vector(map(Vector, eachrow(df[!, Not(:concept)])))
end
