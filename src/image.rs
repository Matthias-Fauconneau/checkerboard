use {vector::{xy, uint2, minmax, MinMax}, image::Image};

#[allow(dead_code)] pub fn copy(target: &mut Image<&mut [u32]>, source: Image<&[u8]>) {
    let size = vector::component_wise_min(source.size, target.size);
    for y in 0..size.y {
        for x in 0..size.x {
            unsafe{(&mut target[xy{x,y}] as *mut _ as *mut std::simd::u8x4).write_unaligned(std::simd::u8x4::splat(source[xy{x,y}]))};
        }
    }
}

pub fn upscale(target: &mut Image<&mut [u32]>, source: Image<&[u16]>) -> (u32, uint2) {
    let MinMax{min, max} = minmax(source.iter().copied()).unwrap();
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [target.size.x, source.size.x] } else { [target.size.y, source.size.y] };
    assert!(num >= den);
    let target_size = source.size*(num/den); // largest integer fit
    let offset = (target.size-target_size)/2;
    let mut target = target.slice_mut(offset, target_size);
    assert!(source.size <= target.size);
    let scale = target.size.x/source.size.x;
    assert!(scale >= 1);
    let stride_factor = target.stride*scale;
    let mut row = target.as_mut_ptr();
    if min < max { for y in 0..source.size.y {
        {
            let mut row = row;
            for x in 0..source.size.x {
                let value = ((source[vector::xy{x,y}]-min) as u32 * 0xFF/(max-min) as u32) as u8;
                let p = value as u32 | (value as u32)<<8 | (value as u32)<<16;
                let p4 = [p; 4];
                {
                    let mut row = row;
                    for _ in 0..scale { unsafe{
                        {
                            let mut row = row;
                            for _ in 0..scale/4 {
                                (row as *mut [u32; 4]).write_unaligned(p4);
                                row = row.add(4);
                            }
                            for _ in scale/4*4..scale {
                                *row = p;
                                row = row.add(1);
                            }
                        }
                        row = row.add(target.stride as usize);
                    }}
                }
                row = unsafe{row.add(scale as usize)};
            }
        }
        row = unsafe{row.add(stride_factor as usize)};
    }}
    (scale, offset)
}

pub fn downscale(target: &mut Image<&mut [u32]>, source: Image<&[u16]>) -> (u32, uint2) {
    let MinMax{min, max} = minmax(source.iter().copied()).unwrap();
    let [min, max] = [min, max].map(|m| m as u32);
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [target.size.x, source.size.x] } else { [target.size.y, source.size.y] };
    let target_size = source.size/((den+num-1)/num); // largest integer downscale
    assert!(target_size <= target.size);
    let offset = (target.size-target_size)/2;
    let target = target.slice_mut(offset, target_size);
    let scale = source.size.x/target.size.x;
    if min < max { match scale {
        4 => {
            //let [min, max] = [*source.iter().min().unwrap() as u16, *source.iter().max().unwrap() as u16].map(|v| (v*factor as u16*factor as u16+14)/15);
            let _3source_stride_u16 = 3*source.stride as usize*std::mem::size_of::<u16>();
            assert_eq!(source.size.x as usize*std::mem::size_of::<u16>()+_3source_stride_u16, 4*source.stride as usize*std::mem::size_of::<u16>());
            unsafe {
                use std::simd::{SimdUint, u8x4, u16x4, u32x4};
                let mut rows4 : [_; 4] = std::array::from_fn(|i| source.as_ptr().add(i*source.stride as usize) as *const u16x4);
                assert!(source.stride == source.size.x);
                let mut target_row = target.as_ptr();
                for _ in 0..target.size.y {
                    for x in 0..target.size.x {
                        (target_row.add(x as usize) as *mut u8x4).write_unaligned(u8x4::splat((((rows4.map(|row4|row4.read().cast::<u32>()).iter().sum::<u32x4>().reduce_sum() / 16)-min)*0xFF/(max-min)) as u8));
                        //(target_row.add(x as usize) as *mut u8x4).write_unaligned(u8x4::splat(((rows4.map(|row4| row4.read().cast::<u16>()).iter().sum::<u16x4>().reduce_sum() - min)*15/(max-min)) as u8));
                        rows4 = rows4.map(|row4| row4.add(1/*4xu8*/));
                    }
                    target_row = target_row.add(target.stride as usize);
                    rows4 = rows4.map(|row4| row4.byte_add(_3source_stride_u16));
                }
            }
        }
        2 => {
            let source_stride_u16 = source.stride as usize*std::mem::size_of::<u16>();
            assert_eq!(source.size.x, source.stride);
            assert_eq!(source.size.x as usize*std::mem::size_of::<u16>()+source_stride_u16, 2*source.stride as usize*std::mem::size_of::<u16>());
            unsafe {
                use std::simd::{SimdUint, u8x4, u16x2, u32x2};
                let mut rows2 : [_; 2] = std::array::from_fn(|i| source.as_ptr().add(i*source.stride as usize) as *const u16x2);
                assert_eq!(source.stride, source.size.x);
                let mut target_row = target.as_ptr();
                for _ in 0..target.size.y {
                    for x in 0..target.size.x {
                        (target_row.add(x as usize) as *mut u8x4).write_unaligned(u8x4::splat((((rows2.map(|row2|row2.read().cast::<u32>()).iter().sum::<u32x2>().reduce_sum() / 4)-min)*0xFF/(max-min)) as u8));
                        rows2 = rows2.map(|row2| row2.add(1/*2xu8*/));
                    }
                    target_row = target_row.add(target.stride as usize);
                    rows2 = rows2.map(|row2| row2.byte_add(source_stride_u16));
                }
            }
        }
        N => {
            let Nminus1_source_stride_u16 = (N-1) as usize*source.stride as usize*std::mem::size_of::<u16>();
            assert_eq!(source.size.x as usize+Nminus1_source_stride_u16, N as usize*source.stride as usize*std::mem::size_of::<u16>());
            unsafe {
                let mut rows = (0..N as usize).map(|i| source.as_ptr().add(i*source.stride as usize)).collect::<Box<_>>();
                assert_eq!(source.stride, source.size.x);
                let mut target_row = target.as_ptr();
                for _ in 0..target.size.y {
                    for x in 0..target.size.x {
                        use std::simd::u8x4;
                        (target_row.add(x as usize) as *mut u8x4).write_unaligned(
                            u8x4::splat(((rows.iter().map(|row|(0..N).map(|dx| row.add(dx as usize).read() as u32).sum::<u32>()).sum::<u32>()/(N*N)-min)*0xFF/(max-min)) as u8));
                        for row in &mut *rows { *row=row.add(N as usize) }
                    }
                    target_row = target_row.add(target.stride as usize);
                    for row in &mut *rows { *row=row.byte_add(Nminus1_source_stride_u16) }
                }
            }
        }
    }}
    (scale, offset)
}

/*use {vector::{xy, uint2}, crate::matrix::{mat3, apply}};
pub fn affine_blit(target: &mut Image<&mut[u8]>, source: Image<&[u8]>, A: mat3) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let p = xy{x: x as f32, y: y as f32};
        let p = apply(A, p);
        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
        target[xy{x, y}] = source[uint2::from(p)];
    }}
}*/