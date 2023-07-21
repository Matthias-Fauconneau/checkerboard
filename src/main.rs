#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;

enum Result {
    Points([vec2; 4]),
    Fit(vec2, Box<[vec2]>, Box<[vec2]>, Image<Box<[uint2]>>, vec2, vec2, Image<Box<[u16]>>)
}
fn refine(source: Image<&[u16]>, mut points: [vec2; 4], R: u32, debug: &'static str) -> Result {
    let mut target = Image::zero(source.size);
    {
        const R : u32 = 3;
        for y in R..source.size.y-R {
            for x in R..source.size.x-R {
                let [p00,p10,p01,p11] = {let r=R as i32;[xy{x:-r,y:-r},xy{x:r,y:-r},xy{x:-r,y:r},xy{x:r,y:r}]}.map(|d|source[(xy{x,y}.signed()+d).unsigned()]);
                let threshold = ([p00,p10,p01,p11].into_iter().map(|u16| u16 as u32).sum::<u32>()/4) as u16;
                if p00<threshold && p11<threshold && p10>threshold && p01>threshold ||
                   p00>threshold && p11>threshold && p10<threshold && p01<threshold {
                    target[xy{x,y}] = (num::abs(p00 as i32 + p11 as i32 - (p10 as i32 + p01 as i32))/2) as u16;
                }
            }
        }
    }
    let image = target;

    for _ in 0..1 {
        let grid = {
            let image = &image;
            Image::from_iter(xy{x:8,y:6},
                (1..=6).map(|y|
                    (1..=8).map(move |x| {
                        let xy{x, y} = xy{x: x as f32/9., y: y as f32/7.};
                        let [A,B,C,D] = points.clone();
                        let p = y*(x*A+(1.-x)*B) + (1.-y)*(x*D+(1.-x)*C);
                        let p = xy{x: p.x.round() as u32, y: p.y.round() as u32,};
                        let (p,_) = (p.y-R..p.y+R).map(|y| (p.x-R..p.x+R).map(move |x| xy{x,y})).flatten().map(|p| (p, image[p])).max_by_key(|(_,v)| *v).unwrap_or((p,0));
                        p
                    }
                    )
                ).flatten()
            )
        };
        let ref ref_grid = grid;
        let rows = (0..grid.size.y).map(|y| (0..grid.size.x).map(move |x| ref_grid[xy{x,y}]));
        let columns = (0..grid.size.x).map(|x| (0..grid.size.y).map(move|y| ref_grid[xy{x,y}]));
        let column = rows.map(|row| row.map(|p| vec2::from(p)).sum::<vec2>()/8.).collect::<Box<_>>();
        let row = columns.map(|column| column.map(|p| vec2::from(p)).sum::<vec2>()/6.).collect::<Box<_>>();
        let row_axis = row.last().unwrap()-row[0];
        let column_axis = column.last().unwrap()-column[0];
        let center = grid.iter().map(|&p| vec2::from(p)).sum::<vec2>()/(8.*6.);
        if debug != "" { return Result::Fit(center, row, column, grid, row_axis, column_axis, image); }
        points = [xy{x:-1./2.,y:-1./2.},xy{x:1./2.,y:-1./2.},xy{x:1./2.,y:1./2.},xy{x:-1./2.,y:1./2.}].map(|xy{x,y}| center + x*row_axis + y*column_axis);
    }
    Result::Points(points)
}

fn main() {
    #[cfg(not(feature="u3v"))] struct NIR;
    #[cfg(feature="u3v")] struct NIR{
        #[allow(dead_code)] camera: cameleon::Camera<cameleon::u3v::ControlHandle, cameleon::u3v::StreamHandle>,
        payload_receiver: cameleon::payload::PayloadReceiver
    }
    #[cfg(not(feature="u3v"))] impl NIR {
        fn new() -> Self { Self }
        fn next(&mut self) -> Image<Box<[u16]>> { panic!("!u3v") }
    }
    #[cfg(feature="u3v")] impl NIR {
        fn new() -> Self {
            let mut cameras = cameleon::u3v::enumerate_cameras().unwrap();
            //for camera in &cameras { println!("{:?}", camera.info()); }
            let mut camera = cameras.remove(0); // find(|c| c.info().contains("U3-368xXLE-NIR")).unwrap()
            camera.open().unwrap();
            camera.load_context().unwrap();
            let mut params_ctxt = camera.params_ctxt().unwrap();
            let acquisition_frame_rate = params_ctxt.node("AcquisitionFrameRate").unwrap().as_float(&params_ctxt).unwrap(); // 30fps
            let max = acquisition_frame_rate.max(&mut params_ctxt).unwrap();
            acquisition_frame_rate.set_value(&mut params_ctxt, max).unwrap(); // 28us
            let exposure_time = params_ctxt.node("ExposureTime").unwrap().as_float(&params_ctxt).unwrap();
            exposure_time.set_value(&mut params_ctxt, 100.).unwrap(); // 15ms=66Hz
            Self{payload_receiver: camera.start_streaming(3).unwrap(), camera}
         }
        fn next(&mut self) -> Image<Box<[u16]>> {
            let payload = self.payload_receiver.recv_blocking().unwrap();
            let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
            let image = Image::new(xy{x: width as u32, y: height as u32}, payload.image().unwrap());
            let image = Image::from_iter(image.size, image.iter().map(|&u8| u8 as u16));
            self.payload_receiver.send_back(payload);
            image
        }
    }
    let nir = /*std::env::args().any(|a| a=="nir").*/true.then(NIR::new);

    #[cfg(not(feature="uvc"))] struct IR;
    #[cfg(feature="uvc")] struct IR(*mut uvc::uvc_stream_handle_t);
    #[cfg(not(feature="uvc"))] impl IR {
        fn new() -> Self { Self }
        fn next(&mut self) -> Image<Box<[u16]>> { panic!("!uvc") }
    }
    #[cfg(feature="uvc")] impl IR {
        fn new() -> Self {
            use std::ptr::null_mut;
            let mut uvc = null_mut();
            use uvc::*;
            assert!(unsafe{uvc_init(&mut uvc as *mut _, null_mut())} >= 0);
            let mut devices : *mut *mut uvc_device_t = null_mut();
            assert!(unsafe{uvc_find_devices(uvc, &mut devices as *mut _, 0/*xbda*/, 0/*x5840*/, std::ptr::null())} >= 0);
            for device in std::iter::successors(Some(devices), |devices| Some(unsafe{devices.add(1)})) {
                let device = unsafe{*device};
                if device.is_null() { break; }
                let mut device_descriptor : *mut uvc_device_descriptor_t = null_mut();
                assert!(unsafe{uvc_get_device_descriptor(device, &mut device_descriptor as &mut _)} >= 0);
                assert!(!device_descriptor.is_null());
                /*let device_descriptor = unsafe{*device_descriptor};
                println!("{:x} {:x} {:x}", device_descriptor.idVendor, device_descriptor.idProduct, device_descriptor.bcdUVC);
                if !device_descriptor.serialNumber.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.serialNumber)}); }
                if !device_descriptor.manufacturer.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.manufacturer)}); }
                if !device_descriptor.product.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.product)}); }*/
                let mut device_handle = null_mut();
                assert!(unsafe{uvc_open(device, &mut device_handle as *mut _)} >= 0);
                let mut control = unsafe{std::mem::zeroed()};
                if unsafe{uvc_get_stream_ctrl_format_size(device_handle, &mut control as *mut _, uvc_frame_format_UVC_FRAME_FORMAT_ANY, 256, 192, 25)} < 0 { continue; }
                let mut stream = null_mut();
                assert!(unsafe{uvc_stream_open_ctrl(device_handle, &mut stream as *mut _, &mut control as *mut _)} >= 0);
                assert!(unsafe{uvc_stream_start(stream, None, null_mut(), 0)} >= 0);
                return Self(stream);
            }
            panic!();
        }
        fn next(&mut self) -> Image<Box<[u16]>> {
            use uvc::*;
            let mut frame : *mut uvc_frame_t = std::ptr::null_mut();
            assert!(unsafe{uvc_stream_get_frame(self.0, &mut frame as *mut _, 1000000)} >= 0);
            assert!(!frame.is_null());
            let uvc_frame_t{width, height, data, data_bytes, ..} = unsafe{*frame};
            Image::new(xy{x: width as u32, y: height as u32}, Box::from(unsafe{std::slice::from_raw_parts(data as *const u16, (data_bytes/2) as usize)}))
        }
    }
    let ir = /*std::env::args().any(|a| a=="ir")*/true.then(IR::new);

    struct View {
        nir: Option<NIR>,
        ir: Option<IR>,
        last_frame: [Option<Image<Box<[u16]>>>; 2],
        debug: &'static str,
        debug_which: &'static str,
    }
    impl ui::Widget for View {
        fn size(&mut self, _: size) -> size { xy{x: 2592, y: 1944} }
        fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
            let nir = self.nir.as_mut().map(|nir| {
                let nir = nir.next();
                self.last_frame[0] = Some(nir.clone());
                nir
            });
            /*#[cfg(feature="png")] {nir = nir.or_else(|| {
                let image = png::open("nir.png").unwrap();
                Image::new(vector::xy{x: image.width(), y: image.height()}, image.into_luma8().into_raw().into_boxed_slice().iter().map(|&u8| u8 as u16).collect())
            });}*/
            let nir = nir.unwrap();
            
            let ir = self.ir.as_mut().map(|ir| {
                let ir = ir.next();
                self.last_frame[1] = Some(ir.clone());
                ir
            }).unwrap_or_else(|| {
                fn cast_slice_box<A,B>(input: Box<[A]>) -> Box<[B]> { // ~bytemuck but allows unequal align size
                    unsafe{Box::<[B]>::from_raw({let len=std::mem::size_of::<A>() * input.len() / std::mem::size_of::<B>(); core::slice::from_raw_parts_mut(Box::into_raw(input) as *mut B, len)})}
                }
                Image::new(xy{x:256,y:192}, cast_slice_box(std::fs::read("ir").unwrap().into_boxed_slice()))
            });

            fn cross(target:&mut Image<&mut[u32]>, scale:f32, offset:uint2, p:vec2, color:u32) {
                let mut plot = |dx,dy| {
                    let Some(p) = (int2::from(scale*p)+xy{x: dx, y: dy}).try_unsigned() else {return};
                    if let Some(p) = target.get_mut(offset+p) { *p = color; }
                };
                for dy in -64..64 { plot(0, dy); }
                for dx in -64..64 { plot(dx, 0); }
            }

            let P_nir = match checkerboard(nir.as_ref(), true, if self.debug_which=="nir" {self.debug} else{""}) {
                checkerboard::Result::Image(image) => { scale(target, image.as_ref()); return Ok(()); }
                /*checkerboard::Result::Points(points) => {
                    let (_, scale, offset) = scale(target, nir.as_ref());
                    for p in points { cross(target, scale, offset, p, 0xFF00FF); }
                    return Ok(());
                }*/
                checkerboard::Result::Points(points, image) => {
                    let (_, scale, offset) = scale(target, image.as_ref());
                    //for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, _) in points { cross(target, scale, offset, p, color); }}
                    for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, neighbours) in points { 
                        cross(target, scale, offset, p, color);
                        //println!("{} {:?}", p, neighbours);
                        let o = vec2::from(offset)+scale*p;
                        for p in neighbours { if let Some(p) = p {for (p,_,_,_) in ui::line::generate_line(target.size, [o,vec2::from(offset)+scale*p]) { target[p] = 0xFFFF; }} }
                    }}
                    return Ok(());
                }
                checkerboard::Result::Checkerboard(points) => {
                    let points = match refine(nir.as_ref(), points, 128, if self.debug_which=="nir" {self.debug} else{""}) {
                        Result::Points(points) => points,
                        Result::Fit(center, row, column, grid, row_axis, column_axis, corner) => {
                            let (_, scale, offset) = scale(target, corner.as_ref());
                            cross(target, scale, offset, center, 0xFFFFFF);
                            for x in 0..grid.size.x {
                                cross(target, scale, offset, row[x as usize], 0x00FF00);
                            }
                            for y in 0..grid.size.y {
                                cross(target, scale, offset, column[y as usize], 0x00FF00);
                                for x in 0..grid.size.x {
                                    cross(target, scale, offset, grid[xy{x,y}].into(), 0x00FFFF);
                                    let xy{x, y} = xy{x: x as f32/7.,  y: y as f32/5.};
                                    //assert!(x >= 0. && x <= 1., "{x}");
                                    //assert!(y >= 0. && y <= 1., "{y}");
                                    let xy{x, y} = xy{x: x-1./2., y: y-1./2.};
                                    //assert!(x >= -1./2. && x <= 1./2., "{x}");
                                    //assert!(y >= -1./2. && y <= 1./2., "{y}");
                                    let p = center + x*row_axis + y*column_axis;
                                    cross(target, scale, offset, p, 0xFF00FF); // purple
                                }
                            }
                            return Ok(());
                        }
                    };
                    if self.debug_which=="nir" && self.debug=="checkerboard" {
                        let (_, scale, offset) = scale(target, nir.as_ref());
                        //for p in points { cross(target, scale, offset, p, u32::MAX); }
                        for (i,&p) in points.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
                        return Ok(());
                    }
                    points
                }
            };

            if self.debug_which=="ir" && ["original","source"].contains(&self.debug) { scale(target, ir.as_ref()); return Ok(()); }

            let source = ir.as_ref();

            // High pass
            let vector::MinMax{min, max} = vector::minmax(source.iter().copied()).unwrap();
            let mut low_then_high = transpose_low_pass_1D::<16>(transpose_low_pass_1D::<16>(source.as_ref()).as_ref());
            if !(min < max) { return Ok(()); }
            low_then_high.as_mut().zip_map(&source, |&low, &p| (0x8000+(((p-min) as u32 *0xFFFF/(max-min) as u32) as i32-low as i32).clamp(-0x8000,0x7FFF)) as u16);
            let source = low_then_high;

            if self.debug_which=="ir" && ["high"].contains(&self.debug) { scale(target, source.as_ref()); return Ok(()); }

            let source = {
                let mut target = Image::zero(source.size);
                {
                    const R : u32 = 3;
                    for y in R..source.size.y-R {
                        for x in R..source.size.x-R {
                            let [p00,p10,p01,p11] = {let r=R as i32;[xy{x:-r,y:-r},xy{x:r,y:-r},xy{x:-r,y:r},xy{x:r,y:r}]}.map(|d|source[(xy{x,y}.signed()+d).unsigned()]);
                            let threshold = ([p00,p10,p01,p11].into_iter().map(|u16| u16 as u32).sum::<u32>()/4) as u16;
                            if p00<threshold && p11<threshold && p10>threshold && p01>threshold ||
                               p00>threshold && p11>threshold && p10<threshold && p01<threshold {
                                target[xy{x,y}] = (num::abs(p00 as i32 + p11 as i32 - (p10 as i32 + p01 as i32))/2) as u16;
                            }
                        }
                    }
                }
                target
            };

            let source = transpose_low_pass_1D::<1>(transpose_low_pass_1D::<1>(source.as_ref()).as_ref());
    
            let mut points = Vec::new();
            //let max = {
                const R : u32 = 12;
                //let mut target = Image::zero(source.size);
                for y in R..source.size.y-R { for x in R..source.size.x-R {
                    let center = source[xy{x, y}];
                    if center < 16384 { continue; }
                    let mut flat = 0;
                    if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                        if (dx,dy) == (0,0) { continue; }
                        if source[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
                        if source[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
                    }} true})() { 
                        //target[xy{x,y}] = center; 
                        points.push(xy{x,y});
                    }
                }}
                //target
            //};
            /*let mut max = max;
            let mut points = Vec::new();
            let R : u32 = 12; // Merges close peaks (top left first)
            for y in R..max.size.y-R { for x in R..max.size.x-R {
                let value = max[xy{x,y}];
                if value == 0 { continue; }
                let mut flat = 0;
                let mut sum : uint2 = num::zero();
                for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                    let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
                    if max[p] == 0 { continue; }
                    flat += 1;
                    sum += p;
                }}
                points.push( (vec2::from(int2::from(xy{x,y}+sum))/((1+flat) as f32), value) );
                for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                    let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
                    max[p] = 0; // Merge small flat plateaus as single point
                }}
            }}*/

            //let points = points.iter().map(|(a,_)| (*a, points.iter().filter(|(b,_)| a!=b).map(|&(b,_)| (b, vector::sq(b-a))).min_by(|(_,a),(_,b)| f32::total_cmp(a, b)).unwrap().0));
            if points.len() < 4 { return Ok(()); }
            let points : Vec<_> = points.iter().map(|&a| (a, {
                //points.iter().filter(|(b,_)| a!=b).map(|&(b,_)| (b, vector::sq(b-a))).min_by(|(_,a),(_,b)| f32::total_cmp(a, b)).unwrap().0
                let mut p = points.iter().filter(|&b| a!=*b).copied().collect::<Box<_>>();
                assert!(p.len() >= 3);
                let closest = if p.len() > 3 { let (closest, _, _) = p.select_nth_unstable_by_key(3, |p| vector::sq(p-a)); &*closest } else { &p };
                <[uint2; 3]>::try_from(closest).unwrap().map(|p| (vector::sq(p-a)<32*32).then_some(p))
            })).collect();

            let graph = |seed| {
                let mut connected = Vec::new();
                fn walk(points: &[(uint2, [Option<uint2>; 3])], mut connected: &mut Vec<uint2>, (p, neighbours): (uint2, [Option<uint2>; 3])) {
                    if !connected.contains(&p) {
                        connected.push(p);
                        for p in neighbours.iter().filter_map(|p| *p) { walk(points, &mut connected, *points.iter().find(|(q,_)| q==&p).unwrap()); }
                    }
                }
                walk(&points, &mut connected, seed);
                connected
            };
            let points = points.iter().map(|&p| graph(p)).max_by_key(|g| g.len()).unwrap();

            //assert!(!points.is_empty());
            if points.len() < 4 || self.debug_which=="ir" /*&& self.debug==""*/ {
                let (_, scale, offset) = scale(target, source.as_ref());
                for a/*(a, neighbours)*/ in points { 
                    cross(target, scale, offset, a.into(), 0xFF00FF); 
                    //for p in neighbours { if let Some(p) = p { for (p,_,_,_) in ui::line::generate_line(target.size, [vec2::from(offset)+scale*vec2::from(a),vec2::from(offset)+scale*vec2::from(p)]) { target[p] = 0xFFFF; } } }
                }
                return Ok(());
            }

            let mut Q : Vec<_> = points.iter().map(|&uint2| vec2::from(uint2)).collect();
            while Q.len() > 4 {
                Q.remove(((0..Q.len()).map(|i| {
                    let [p0, p1, p2]  = std::array::from_fn(|j| Q[(i+j)%Q.len()]);
                    (vector::cross2(p2-p0, p1-p0), i)
                }).min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap().1+1)%Q.len());
            }
            let i0 = Q.iter().enumerate().min_by(|(_,a),(_,b)| (a.x+a.y).total_cmp(&(b.x+b.y))).unwrap().0; // top left
            let mut Q = [0,1,2,3].map(|i|Q[(i0+i)%4]);
        
            // First edge is long edge
            use vector::norm;
            if norm(Q[2]-Q[1])+norm(Q[0]-Q[3]) > norm(Q[1]-Q[0])+norm(Q[3]-Q[2]) { Q.swap(1,3); }
        
            let P_ir = Q;
        
            /*let P_ir = match checkerboard(ir.as_ref(), false, if self.debug_which=="ir" {self.debug} else{""}) {
                checkerboard::Result::Image(image) => { scale(target, image.as_ref()); return Ok(()); }
                /*checkerboard::Result::Points(points) => {
                    let (_, scale, offset) = scale(target, ir.as_ref());
                    //assert!(!points.is_empty());
                    for p in points { cross(target, scale, offset, p, 0xFF00FF); }
                    return Ok(());
                }*/
                checkerboard::Result::Points(points, image) => {
                    let (_, scale, offset) = scale(target, image.as_ref());
                    for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, neighbours) in points { 
                        cross(target, scale, offset, p, color);
                        println!("{} {:?}", p, neighbours);
                        let o = vec2::from(offset)+scale*p;
                        for p in neighbours { for (p,_,_,_) in ui::line::generate_line(target.size, [o,vec2::from(offset)+scale*p.unwrap()]) { target[p] = 0xFFFF; } }
                    }}
                    return Ok(());
                }
                //checkerboard::Result::Checkerboard(points) => {
                checkerboard::Result::Checkerboard(points) => {
                    let points = match refine(ir.as_ref(), points, 6, if self.debug_which=="ir" {self.debug} else{""}) {
                        Result::Points(points) => points,
                        Result::Fit(center, row, column, grid, row_axis, column_axis, corner) => {
                            let (_, scale, offset) = scale(target, corner.as_ref());
                            cross(target, scale, offset, center, 0xFFFFFF);
                            for x in 0..grid.size.x {
                                cross(target, scale, offset, row[x as usize], 0x00FF00);
                            }
                            for y in 0..grid.size.y {
                                cross(target, scale, offset, column[y as usize], 0x00FF00);
                                for x in 0..grid.size.x {
                                    cross(target, scale, offset, grid[xy{x,y}].into(), 0x00FFFF);
                                    let xy{x, y} = xy{x: x as f32/7.,  y: y as f32/5.};
                                    //assert!(x >= 0. && x <= 1., "{x}");
                                    //assert!(y >= 0. && y <= 1., "{y}");
                                    let xy{x, y} = xy{x: x-1./2., y: y-1./2.};
                                    //assert!(x >= -1./2. && x <= 1./2., "{x}");
                                    //assert!(y >= -1./2. && y <= 1./2., "{y}");
                                    let p = center + x*row_axis + y*column_axis;
                                    cross(target, scale, offset, p, 0xFF00FF); // purple
                                }
                            }
                            return Ok(());
                        }
                    };
                    if self.debug_which=="ir" && self.debug=="checkerboard" {
                        //let (_, scale, offset) = scale(target, ir.as_ref());
                        let (_, scale, offset) = scale(target, ir.as_ref());
                        for p in points { cross(target, scale, offset, p, 0xFF00FF); } // purple
                        //for (i,&p) in points.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF0][i]); }    
                        //let (_, NIR_scale, NIR_offset) = scale(target, nir.as_ref()); // FIXME
                        //for p in P_nir { cross(target, NIR_scale, NIR_offset, p, 0x00FF00); } //green
                        //for (i,&p) in P_nir.iter().enumerate() { cross(target, scale, offset, p, [0x00_00FF,0x00_FF00,0x00_FFFF,0xFF_0000][i]); }
                        
                        return Ok(());
                    }
                    points
                }
            };*/

            /*#[cfg(feature="opencv")] let P_ir = {
                let mut corners = opencv::core::Mat::default();
                if !{
                    let ir = Image::from_iter(ir.size, ir.iter().map(|&u16| u16 as u8));
                    opencv::calib3d::find_chessboard_corners(&opencv::core::Mat::from_slice_rows_cols(&ir, ir.size.x as usize, ir.size.y as usize).unwrap(), opencv::core::Size::new(8,6), &mut corners, 0).unwrap() 
                } {
                    scale(target, ir.as_ref());
                    return Ok(());
                }
                panic!("{corners:?}")
            };*/

            let P = [P_nir, P_ir]; //P[1] = [P[1][0], P[1][3], P[1][1], P[1][2]];
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            let A = mul(inverse(M[1]), mul(A, M[0]));

            let (_, IR_scale, IR_offset) = scale(target, ir.as_ref()); // FIXME
            let (target_size, scale, offset) = scale(target, nir.as_ref());
            /*for (i,&p) in P_nir.iter().enumerate() { cross(target, scale, offset, p, [0x00_0000,0x00_00FF,0x00_FF00,0x00_FFFF][i]); }    
            for (i,&p) in P_ir.iter().enumerate() { cross(target, IR_scale, IR_offset, p, [0xFF_0000,0xFF_00FF,0xFF_FF00,0xFF_FFFFF][i]); }*/
            for (i,&p) in P_nir.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
            for (i,&p) in P_ir.iter().enumerate() { cross(target, IR_scale, IR_offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }
            affine_blit(target, target_size, ir.as_ref(), A, nir.size);
            Ok(())
        }
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, event: &ui::Event) -> ui::Result<bool> {
            use ui::Event::Key;
            match event {
                Key('a') =>{ self.debug = ""; return Ok(true); }
                Key('b') =>{ self.debug = "binary"; return Ok(true); }
                Key('c') =>{ self.debug = "contour"; return Ok(true); }
                Key('d') =>{ self.debug = "distance"; return Ok(true); }
                Key('e') =>{ self.debug = "erode"; return Ok(true); }
                Key('f') =>{ self.debug = "filtered"; return Ok(true); }
                Key('h') =>{ self.debug = "high"; return Ok(true); }
                Key('i') =>{ self.debug_which = "ir"; return Ok(true); }
                Key('l') =>{ self.debug = "low"; return Ok(true); }
                Key('m') =>{ self.debug = "max"; return Ok(true); }
                Key('n') =>{ self.debug_which = "nir"; return Ok(true); }
                Key('o') =>{ self.debug = "original"; return Ok(true); }
                Key(' ') =>{ self.debug = "checkerboard"; return Ok(true); }
                Key('\n') => { self.debug_which=""; return Ok(true); }
                Key('p') =>{ self.debug = "peaks"; return Ok(true); }
                Key('q') =>{ self.debug = "quads"; return Ok(true); }
                //Key('s') =>{ self.debug = "selection"; return Ok(true); }
                Key('s') =>{ self.debug = "source"; return Ok(true); }
                Key('⎙')|Key('\u{F70C}')/*|Key(' ')*/ => {
                    println!("⎙");
                    if self.last_frame.iter_mut().zip(["nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| std::fs::write(name, &bytemuck::cast_slice(image)).unwrap()).count() == 0 {
                        self.ir = Some(IR::new());
                    }
                    /*#[cfg(feature="png")] {
                        let mut target = Image::uninitialized(xy{x: 2592, y:1944});
                        let size = target.size;
                        self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap();
                        png::save_buffer("checkerboard.png", bytemuck::cast_slice(&target.data), target.size.x, target.size.y, png::ColorType::Rgba8).unwrap();
                    }*/
                },
                _ => {},
            }
            Ok(/*self.nir.is_some()||self.ir.is_some()*/true)
        }
    }
    ui::run("Checkerboard", &mut View{nir, ir, last_frame: [None, None], debug:"", debug_which: "nir"}).unwrap();
}
