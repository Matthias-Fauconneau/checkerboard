use {vector::xy, image::Image};

#[allow(dead_code)] pub fn copy(target: &mut Image<&mut [u32]>, source: Image<&[u8]>) {
    let size = vector::component_wise_min(source.size, target.size);
    for y in 0..size.y {
        for x in 0..size.x {
            unsafe{(&mut target[xy{x,y}] as *mut _ as *mut std::simd::u8x4).write_unaligned(std::simd::u8x4::splat(source[xy{x,y}]))};
        }
    }
}

pub fn upscale(target: &mut Image<&mut [u32]>, source: Image<&[u16]>) {
    let [min, max] = [*source.iter().min().unwrap(), *source.iter().max().unwrap()];
    if !(min < max) { return; }
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [target.size.x, source.size.x] } else { [target.size.y, source.size.y] };
    let target_size = source.size*(num/den); // largest integer fit
    let mut target = target.slice_mut((target.size-target_size)/2, target_size);
    let factor = target.size.x/source.size.x;
    assert!(factor >= 1);
    let stride_factor = target.stride*factor;
    let mut row = target.as_mut_ptr();
    for y in 0..source.size.y {
        {
            let mut row = row;
            for x in 0..source.size.x {
                let value = ((source[vector::xy{x,y}]-min) as u32 * 0xFF/(max-min) as u32) as u8;
                let p = value as u32 | (value as u32)<<8 | (value as u32)<<16;
                let p4 = [p; 4];
                {
                    let mut row = row;
                    for _ in 0..factor { unsafe{
                        {
                            let mut row = row;
                            for _ in 0..factor/4 {
                                (row as *mut [u32; 4]).write_unaligned(p4);
                                row = row.add(4);
                            }
                            for _ in factor/4*4..factor {
                                *row = p;
                                row = row.add(1);
                            }
                        }
                        row = row.add(target.stride as usize);
                    }}
                }
                row = unsafe{row.add(factor as usize)};
            }
        }
        row = unsafe{row.add(stride_factor as usize)};
    }
}

/*pub fn downscale(target: &mut Image<&mut [u32]>, source: Image<&[u8]>) {
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [target.size.x, source.size.x] } else { [target.size.y, source.size.y] };
    let target_size = source.size/((den+num-1)/num); // largest integer downscale
    assert!(target_size <= target.size);
    let target = target.slice_mut((target.size-target_size)/2, target_size);
    let factor = source.size.x/target.size.x;
    assert_eq!(factor, 4, "{source:?} {num} {den} {target:?} {factor}");
    //let [min, max] = [*source.iter().min().unwrap() as u16, *source.iter().max().unwrap() as u16].map(|v| (v*factor as u16*factor as u16+14)/15);
    let _3source_stride = 3*source.stride as usize;
    assert_eq!(source.size.x+_3source_stride as u32, 4*source.stride);
    unsafe {
        use std::simd::{SimdUint, u8x4, u16x4};
        let mut rows4 : [_; 4] = std::array::from_fn(|i| source.as_ptr().add(i*source.stride as usize) as *const u8x4);
        assert!(source.stride == source.size.x);
        let mut target_row = target.as_ptr();
        for _ in 0..target.size.y { 
            for x in 0..target.size.x { 
                (target_row.add(x as usize) as *mut u8x4).write_unaligned(u8x4::splat((rows4.map(|row4| row4.read().cast::<u16>()).iter().sum::<u16x4>().reduce_sum() / 16) as u8));
                //(target_row.add(x as usize) as *mut u8x4).write_unaligned(u8x4::splat(((rows4.map(|row4| row4.read().cast::<u16>()).iter().sum::<u16x4>().reduce_sum() - min)*15/(max-min)) as u8));
                rows4 = rows4.map(|row4| row4.add(1/*4xu8*/));
            }
            target_row = target_row.add(target.stride as usize);
            rows4 = rows4.map(|row4| row4.byte_add(_3source_stride));
        }
    }
}*/

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