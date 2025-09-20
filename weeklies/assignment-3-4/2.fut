-- cum_sums_of_squares.fut
-- For each row (length 64), return the inclusive prefix sums of squares.
--
-- ==
-- compiled input {
--   [[1.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, -6.0, 1.0,
--     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
--     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
--     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
--     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
--     0.0, 0.0, 0.0, 0.0]]
-- }
-- output {
--   [[1.0, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 18.0, 34.0, 70.0, 71.0,
--     71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0,
--     71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0,
--     71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0,
--     71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0, 71.0,
--     71.0, 71.0, 71.0, 71.0]]
-- }

-- Row-level kernel: take a 1D row, return a 1D row
let row_prefix_sums_of_squares_par [m] (row: [m]f64) : [m]f64 =
  let squared : [m]f64 = map (\x -> x * x) row
  in scan (+) 0.0f64 squared

-- Matrix-level: map the row kernel across rows
let matrix_prefix_sums_of_squares_par [n][m] (A: [n][m]f64) : [n][m]f64 =
  map row_prefix_sums_of_squares_par A

-- Entry: expects [n][64]f64, returns [n][64]f64
entry main [n] (a: [n][64]f64) : [n][64]f64 =
  matrix_prefix_sums_of_squares_par a