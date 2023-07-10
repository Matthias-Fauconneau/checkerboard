#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit)]#![allow(non_camel_case_types,non_snake_case)]
use {vector::{xy, size, int2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
//mod matrix; use matrix::*;
mod image; use image::*;

fn main() {
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
    let nir = camera.start_streaming(3).unwrap();
    struct View {
        nir: cameleon::payload::PayloadReceiver,
    }
    impl ui::Widget for View { 
        fn size(&mut self, _: size) -> size { xy{x: 2592, y: 1944} }
        #[ui::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { 
            let payload = self.nir.recv_blocking().unwrap();
            let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
            let nir = Image::new(xy{x: width as u32, y: height as u32}, payload.image().unwrap());
            //let nir = Image::new(xy{x: width as u32, y: height as u32}, neg(payload.image().unwrap()));
            //let (_checkerboard, (binary, debug)) = checkerboard(nir.as_ref());
            copy(target, nir.as_ref());
            //copy(target, binary.as_ref());
            /*for (i, points) in debug.into_iter().enumerate() {
                for (p,_) in points { 
                    let mut plot = |x,y| {
                        let Some(p) = (int2::from(p)+xy{x,y}).try_unsigned() else { return; };
                        if let Some(target) = target.get_mut(p) { *target = if i>0 { ui::color::red.into() } else { ui::color::blue.into() }; }
                    };
                    for y in -16..16 { plot(0, y); }
                    for x in -16..16 { plot(x, 0); }
                }
            }*/
            if let Some(checkerboard) = checkerboard(nir.as_ref()) {
                println!("{checkerboard:?}");
                for &p in &checkerboard { 
                    let mut plot = |x,y| if let Some(p) = target.get_mut((int2::from(p)+xy{x,y}).unsigned()) { *p = bgr8::from(0xFFu8).into(); };
                    for y in -16..16 { plot(0, y); }
                    for x in -16..16 { plot(x, 0); }
                }
            }
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

            self.nir.send_back(payload);
        }
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, _: &ui::Event) -> ui::Result<bool> { Ok(true) }
    }
    ui::run("Checkerboard", &mut View{nir}).unwrap();
}
