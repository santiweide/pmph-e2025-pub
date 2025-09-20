-- For each row, compute inclusive prefix sums of squares.
entry cum_sums_of_squares (a: [n][64]f32) : [n][64]f32 =
  map (\(row: [64]f32) ->
         let squares : [64]f32 = map (\x -> x * x) row
         in scan (+) 0.0f32 squares)   -- inclusive: [x0^2, x0^2+x1^2, ...]
      a

