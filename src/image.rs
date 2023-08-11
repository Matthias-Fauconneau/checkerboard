use {vector::{xy, uint2, int2, vec2, minmax, MinMax}, image::Image};

pub fn upscale(target: &mut Image<&mut [u32]>, source: Image<&[u16]>) -> (uint2, u32, uint2) {
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
                let p = value as u32 | (value as u32)<<8 | (value as u32)<<16 | (0xFF<<24);
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
    (target_size, scale, offset)
}

pub fn downscale(target: &mut Image<&mut [u32]>, source: Image<&[u16]>) -> (uint2, u32, uint2) {
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
            let _3source_stride_u16 = (4*source.stride-source.size.x) as usize*std::mem::size_of::<u16>();
            assert_eq!(source.size.x as usize*std::mem::size_of::<u16>()+_3source_stride_u16, 4*source.stride as usize*std::mem::size_of::<u16>());
            unsafe {
                use std::simd::{SimdUint, u8x4, u16x4, u32x4};
                let mut rows4 : [_; 4] = std::array::from_fn(|i| source.as_ptr().add(i*source.stride as usize) as *const u16x4);
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
            let source_stride_u16 = (2*source.stride-source.size.x) as usize*std::mem::size_of::<u16>();
            assert_eq!(source.size.x as usize*std::mem::size_of::<u16>()+source_stride_u16, 2*source.stride as usize*std::mem::size_of::<u16>());
            unsafe {
                use std::simd::{SimdUint, u8x4, u16x2, u32x2};
                let mut rows2 : [_; 2] = std::array::from_fn(|i| source.as_ptr().add(i*source.stride as usize) as *const u16x2);
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
    (target_size, scale, offset)
}

pub fn scale(target: &mut Image<&mut[u32]>, image: Image<&[u16]>) -> (uint2, f32, uint2) {
    if image.size <= target.size { let (target_size, scale, offset) = upscale(target, image); (target_size, scale as f32, offset) }
    else { let (target_size, scale, offset) = downscale(target, image); (target_size, 1./scale as f32, offset) }
}

use image::bgr8;

use crate::matrix::{mat3, apply};
pub fn affine_blit(target: &mut Image<&mut[u32]>, fit_size: uint2, source: Image<&[u16]>, A: mat3, transform_target_size: uint2, _i: usize) -> (f32, uint2) {
    assert!(fit_size <= target.size);
    let offset = (target.size-fit_size)/2;
    let scale = transform_target_size.x as f32/fit_size.x as f32;
    let mut target = target.slice_mut(offset, fit_size);
    let MinMax{min, max} = minmax(source.iter().copied()).unwrap();
    for y in 0..target.size.y { for x in 0..target.size.x {
        let p = scale*xy{x: x as f32, y: y as f32};
        let p = apply(A, p);
        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
        let s = source[uint2::from(p)];
        let p = &mut target[xy{x, y}];
        //*p = {let mut p = <[_; _]>::from(bgr8::from(*p)); p[i] = ((s-min) as u32*0xFF/(max-min) as u32) as u8; bgr8::from(p)}.into();
        *p = u32::from(bgr8::from(((s-min) as u32*0xFF/(max-min) as u32) as u8));
    }}
    (scale, offset)
}

pub fn write_raw(name: &str, image: Image<&[u16]>) { std::fs::write(format!("{name}.{}x{}",image.size.x, image.size.y), &bytemuck::cast_slice(&image)).unwrap() }
fn cast_slice_box<A,B>(input: Box<[A]>) -> Box<[B]> { // ~bytemuck but allows unequal align size
    unsafe{Box::<[B]>::from_raw({let len=std::mem::size_of::<A>() * input.len() / std::mem::size_of::<B>(); core::slice::from_raw_parts_mut(Box::into_raw(input) as *mut B, len)})}
}
pub fn raw(name: &str, size: uint2) -> Option<Image<Box<[u16]>>> { Some(Image::new(size, cast_slice_box(std::fs::read(format!("{name}.{}x{}",size.x,size.y)).ok()?.into_boxed_slice()))) }

pub fn write(path: impl AsRef<std::path::Path>, target: Image<&[u32]>) {
    #[cfg(not(feature="png"))] unimplemented!("{:?} {target:?}", path.as_ref());
    #[cfg(feature="png")] png::save_buffer(path, bytemuck::cast_slice(&target.data), target.size.x, target.size.y, png::ColorType::Rgba8).unwrap();
}
/*pub fn open(path: impl AsRef<std::path::Path>) -> Image<Box<[u16]>> {
    #[cfg(not(feature="png"))] unimplemented!();
    #[cfg(feature="png")] {
        let image = png::open(path).unwrap();
        Image::new(vector::xy{x: image.width(), y: image.height()}, image.into_luma8().into_raw().into_boxed_slice().iter().map(|&u8| u8 as u16).collect())
    }
}*/
