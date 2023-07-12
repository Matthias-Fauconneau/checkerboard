#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer)]#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, size, int2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
//mod matrix; use matrix::*;
mod image; use image::*;

fn main() {
    let nir = false.then(|| {
        let mut cameras = cameleon::u3v::enumerate_cameras().unwrap();
        //for camera in &cameras { println!("{:?}", camera.info()); }
        let ref mut camera = cameras[0]; // find(|c| c.info().contains("U3-368xXLE-NIR")).unwrap()
        camera.open().unwrap();
        camera.load_context().unwrap();          
        let mut params_ctxt = camera.params_ctxt().unwrap();
        let acquisition_frame_rate = params_ctxt.node("AcquisitionFrameRate").unwrap().as_float(&params_ctxt).unwrap(); // 30fps
        let max = acquisition_frame_rate.max(&mut params_ctxt).unwrap();
        acquisition_frame_rate.set_value(&mut params_ctxt, max).unwrap(); // 28us
        let exposure_time = params_ctxt.node("ExposureTime").unwrap().as_float(&params_ctxt).unwrap();
        exposure_time.set_value(&mut params_ctxt, 1000.).unwrap(); // 15ms=66Hz
        camera.start_streaming(3).unwrap()
    });

    type IR = *mut uvc::uvc_stream_handle_t;
    fn IR() -> IR {
        use std::ptr::null_mut;
        let mut uvc = null_mut();
        use uvc::*;
        assert!(unsafe{uvc_init(&mut uvc as *mut _, null_mut())} >= 0);
        let mut devices : *mut *mut uvc_device_t = null_mut();
        assert!(unsafe{uvc_find_devices(uvc, &mut devices as *mut _, 0xbda, 0x5840, std::ptr::null())} >= 0);
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
            return stream;
        }
        panic!();
    }
    let ir = std::env::args().any(|a| a=="ir").then(IR);

    struct View {
        #[allow(dead_code)] nir: Option<cameleon::payload::PayloadReceiver>,
        ir: Option<IR>,
        last_frame: Option<Image<Box<[u16]>>>,
        last_key: char,
        toggle: bool,
    }
    impl ui::Widget for View { 
        fn size(&mut self, _: size) -> size { xy{x: 2592, y: 1944} }
        #[ui::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) {
            /*let payload = self.nir.recv_blocking().unwrap();
            let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
            let nir = Image::new(xy{x: width as u32, y: height as u32}, payload.image().unwrap());*/
            
            let mut data = None;
            let ir = self.ir.map(|ir| {
                use uvc::*;
                let mut frame : *mut uvc_frame_t = std::ptr::null_mut();
                assert!(unsafe{uvc_stream_get_frame(ir, &mut frame as *mut _, 1000000)} >= 0);
                assert!(!frame.is_null());
                //let frame = unsafe{*frame}; 
                let uvc_frame_t{width, height, data, data_bytes, ..} = unsafe{*frame}; 
                let ir = Image::new(xy{x: width as u32, y: height as u32}, unsafe{std::slice::from_raw_parts(data as *const u16, (data_bytes/2) as usize)});
                self.last_frame = Some(ir.clone());
                ir
            }).unwrap_or_else(|| {
                data = Some(std::fs::read("ir").unwrap());
                Image::new(xy{x:256,y:192}, bytemuck::cast_slice(data.as_ref().unwrap()))
            });
            //let ir = Image::new(xy{x: frame.width as u32, y: frame.height as u32}, unsafe{std::slice::from_raw_parts(frame.data as *const u16, (frame.data_bytes/2) as usize)};  
            
            /*match self.last_key {
                'b' => {
                    let binary = checkerboard(ir.as_ref()).unwrap_err();
                    upscale(target, binary.as_ref());
                }
                'o'|_ => upscale(target, ir.as_ref()),
            }*/
            //copy(target, nir.as_ref());          

            let checkerboard::Result::Points(points, distance) = checkerboard(ir.as_ref(), self.toggle) else { unimplemented!() };
            let (factor, offset) = upscale(target, distance.as_ref());
            for (points, &color) in points.iter().zip(&[u32::MAX, 0]) { for &(p, _) in points { 
                let mut plot = |x,y| {
                    let Some(p) = (int2::from(factor as f32*p)+xy{x,y}).try_unsigned() else {return};
                    if let Some(p) = target.get_mut(offset+p) { *p = color; }
                };
                for y in -16..16 { plot(0, y); }
                for x in -16..16 { plot(x, 0); }
            }}

            /*if let Some(checkerboard) = checkerboard(nir.as_ref()) {
                //println!("{checkerboard:?}");
                for &p in &checkerboard { 
                    let mut plot = |x,y| if let Some(p) = target.get_mut((int2::from(p)+xy{x,y}).unsigned()) { *p = bgr8::from(0xFFu8).into(); };
                    for y in -16..16 { plot(0, y); }
                    for x in -16..16 { plot(x, 0); }
                }
            }*/
            /*let mut P = checkerboards; P[1] = [P[1][0], P[1][3], P[1][1], P[1][2]];
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            let A = mul(inverse(M[1]), mul(A, M[0]));
            fn collect<I:Iterator, const N: usize>(mut iter: I) -> [I::Item; N] { let array = [(); N].map(|_| iter.next().unwrap()); assert!(iter.next().is_none()); array }
            let images : [_; 2] = collect(images.iter().zip([(identity(), P[0]), (A, P[1].map(|p| apply(inverse(A), p)))]).map(|(image, (A, P))| {
                let mut target = Image::zero(image.size);
                affine_blit(&mut target.as_mut(), image.as_ref(), A);
                let image = target;

                let max = *image.iter().max().unwrap();
                let mut target = image;
                //target.as_mut().map(|&v| (v as u16 * 0xFF / max as u16) as u8);
                Image::map(&mut target.as_mut(), |&v| (v as u16 * 0xFF / max as u16) as u8);

                for &p in &P {
                    use vector::{uint2, xy};
                    target[uint2::from(/*p.0.round()*/xy{x: p.x.round() as f32, y: p.y.round() as f32})] = 0xFF;
                }
                target
            }));*/

            //self.nir.send_back(payload);
        }
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, event: &ui::Event) -> ui::Result<bool> {
            if let ui::Event::Key('s') = event { 
                if self.ir.is_some() { std::fs::write("ir", &bytemuck::cast_slice(self.last_frame.as_ref().unwrap())).unwrap(); }
                else { self.ir = Some(IR()); }
            }
            if let &ui::Event::Key(' ') = event { self.toggle = !self.toggle; return Ok(true); }
            if let &ui::Event::Key(key) = event { self.last_key = key; return Ok(true); }
            Ok(self.nir.is_some()||self.ir.is_some()) 
        }
    }
    ui::run("Checkerboard", &mut View{nir, ir, last_frame: None, last_key: '\0', toggle: false}).unwrap();
}
