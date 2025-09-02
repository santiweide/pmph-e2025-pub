-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
-- compiled input {
--    [0, 0, 1, 0, 0, 0, 2]
-- }  
-- output { 
--    3
-- }
-- compiled input {
--    [1, 2, 3]
-- }  
-- output { 
--    0
-- }
-- compiled input {
--    [0]
-- }  
-- output { 
--    1
-- }
-- compiled input {
--    [0, 0, 0, 0, 0]
-- }  
-- output { 
--    5
-- }
-- compiled input {
--    empty([0]i32)
-- }  
-- output { 
--    0
-- }
--
-- "Zero-array-1e7" script input { mk_input_zero 10000000i64 }
-- output { 0i32 }
--
-- "Random-input-1e8" compiled random input { [100000000]i32 } auto output

entry mk_input_zero (n:i64) : [n]i32 =
  replicate n (-1i32)

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
