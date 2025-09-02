-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
-- compiled input {
--    [1, 2, 2, 1, 3, 4]
-- }  
-- output { 
--    3
-- }
-- compiled input {
--    [1]
-- }  
-- output { 
--    1
-- }
-- compiled input {
--    empty([0]i32)
-- }  
-- output { 
--    0
-- }
--
-- "Sorted-array-1e6" script input { mk_input_sorted 1000000i64 }
-- output { 1000000 }
-- "Sorted-array-1e8" script input { mk_input_sorted 100000000i64 }
-- output { 100000000 }
--
-- "Random-1e8" compiled random input { [100000000]i32 } auto output

entry mk_input_sorted (n:i64) : [n]i32 =
  iota n |> map (\x -> i32.i64 x+1) -- [1..n]

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs