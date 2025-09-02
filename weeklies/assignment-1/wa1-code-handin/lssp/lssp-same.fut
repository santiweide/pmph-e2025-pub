-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input {
--    [2, 2, 2, 1, 1, 3, 3, 3, 3, 0]
-- }  
-- output { 
--    4
-- }
-- compiled input {
--    [1, 1, 1, 1]
-- }  
-- output { 
--    4
-- }
-- compiled input {
--    [1, 2, 3, 4]
-- }  
-- output { 
--    1
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
-- "Same-array-1e7" script input { mk_input_same 10000000i64 }
-- output { 10000000i32 }
--
-- "Random-input-1e8" compiled random input { [100000000]i32 } auto output

entry mk_input_same (n:i64) : [n]i32 =
  replicate n 1i32

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
