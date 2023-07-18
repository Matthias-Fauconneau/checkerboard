#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;

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
            exposure_time.set_value(&mut params_ctxt, 1000.).unwrap(); // 15ms=66Hz
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
                let device_descriptor = unsafe{*device_descriptor};
                println!("{:x} {:x} {:x}", device_descriptor.idVendor, device_descriptor.idProduct, device_descriptor.bcdUVC);
                if !device_descriptor.serialNumber.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.serialNumber)}); }
                if !device_descriptor.manufacturer.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.manufacturer)}); }
                if !device_descriptor.product.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.product)}); }
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
    let ir = Some(IR::new()); //std::env::args().any(|a| a=="ir").then(IR::new);

    struct View {
        nir: Option<NIR>,
        ir: Option<IR>,
        last_frame: [Option<Image<Box<[u16]>>>; 2],
        debug: &'static str,
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
                for dy in -16..16 { plot(0, dy); }
                for dx in -16..16 { plot(dx, 0); }
            }

            let P_nir = match checkerboard(nir.as_ref(), true, false/*self.toggle*/) {
                checkerboard::Result::Image(image) => { scale(target, image.as_ref()); return Ok(()); }
                checkerboard::Result::Points(points, image) => {
                    let (_, scale, offset) = scale(target, image.as_ref());
                    for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, _) in points { cross(target, scale, offset, p, color); }}
                    return Ok(());
                }
                checkerboard::Result::Checkerboard(points) => {
                    if self.debug=="nir" {
                        let (_, scale, offset) = scale(target, nir.as_ref());
                        for p in points { cross(target, scale, offset, p, u32::MAX); }
                        return Ok(());
                    }
                    points
                }
            };

            let P_ir = match checkerboard(ir.as_ref(), false, false/*self.toggle*/) {
                checkerboard::Result::Image(image) => { scale(target, image.as_ref()); return Ok(()); }
                checkerboard::Result::Points(points, image) => {
                    let (_, scale, offset) = scale(target, image.as_ref());
                    for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, _) in points { cross(target, scale, offset, p, color); }}
                    return Ok(());
                }
                checkerboard::Result::Checkerboard(points) => {
                    if self.debug=="ir" {
                        let (_, scale, offset) = scale(target, ir.as_ref());
                        for p in points { cross(target, scale, offset, p, 0); }
                        return Ok(());
                    }
                    points
                }
            };

            let P = [P_nir, P_ir]; //P[1] = [P[1][0], P[1][3], P[1][1], P[1][2]];
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            let A = mul(inverse(M[1]), mul(A, M[0]));

            let (target_size, _, _) = scale(target, nir.as_ref());
            affine_blit(target, target_size, ir.as_ref(), A, nir.size);
            Ok(())
        }
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, event: &ui::Event) -> ui::Result<bool> {
            use ui::Event::Key;
            match event {
                Key('s') => {
                    if self.last_frame.iter_mut().zip(["nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| std::fs::write(name, &bytemuck::cast_slice(image)).unwrap()).count() == 0 {
                        self.ir = Some(IR::new());
                    }
                },
                Key('i') =>{ self.debug = "ir"; return Ok(true); }
                Key('n') =>{ self.debug = "nir"; return Ok(true); }
                Key('o')|Key(' ')|Key('\n') =>{ self.debug = ""; return Ok(true); }
                //Key(' ') =>{ self.toggle = !self.toggle; return Ok(true); }
                Key('⎙')|Key('\u{F70C}')/*|Key(' ')*/ => {
                    println!("⎙");
                    let mut target = Image::uninitialized(xy{x: 2592, y:1944});
                    let size = target.size;
                    self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap();
                    #[cfg(feature="png")] png::save_buffer("checkerboard.png", bytemuck::cast_slice(&target.data), target.size.x, target.size.y, png::ColorType::Rgba8).unwrap();
                },
                _ => {},
            }
            Ok(/*self.nir.is_some()||self.ir.is_some()*/true)
        }
    }
    ui::run("Checkerboard", &mut View{nir, ir, last_frame: [None, None], /*toggle: true*/debug:""}).unwrap();
}
