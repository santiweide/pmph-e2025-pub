futhark test --backend=cuda lssp-sorted.fut
futhark test --backend=cuda lssp-zeros.fut
futhark test --backend=cuda lssp-same.fut


futhark bench --backend=cuda lssp-sorted.fut
futhark bench --backend=cuda lssp-zeros.fut
futhark bench --backend=cuda lssp-same.fut


futhark bench --backend=c lssp-sorted.fut
futhark bench --backend=c lssp-zeros.fut
futhark bench --backend=c lssp-same.fut
futhark bench --backend=c lssp-seq.fut


Generate test data:
futhark dataset --i32-bounds=-1000:1000 -b -g [100000000]i32 > lssp-data.in
futhark test --backend=c lssp-seq.fut # gets lssp-data.out


futhark bench --backend=cuda spMVmult-seq.fut
futhark bench --backend=cuda spMVmult-flat.fut

futhark bench --backend=c spMVmult-seq.fut
futhark bench --backend=c spMVmult-flat.fut

futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 > data.in
futhark test --backend=c spMVmult-seq.fut

futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./spMVmult-flat -t /dev/stderr -r 10 > /dev/null


futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./spMVmult-seq > /dev/null


futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 > data.1e4.in
futhark run spMVmult-seq.fut < data.1e4.in > data.1e4.out
futhark dataset --i64-bounds=0:9999 -g [10000000]i64 --f32-bounds=-7.0:7.0 -g [10000000]f32 --i64-bounds=100:100 -g [100000]i64 --f32-bounds=-10.0:10.0 -g [100000]f32 > data.1e5.in
futhark run spMVmult-seq.fut < data.1e5.in > data.1e5.out



futhark bench --backend=c spMVmult-seq.fut
futhark bench --backend=cuda spMVmult-flat.fut
