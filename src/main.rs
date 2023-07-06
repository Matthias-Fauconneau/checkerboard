#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit)]#![allow(non_camel_case_types,non_snake_case)]
use {vector::{xy/*, vec2*/}, ::image::Image};
//mod checkerboard; use checkerboard::*;
//mod matrix; use matrix::*;
mod image; use image::*;

fn main() {
    let mut cameras = cameleon::u3v::enumerate_cameras().unwrap();
    for camera in &cameras { println!("{:?}", camera.info()); }
    let ref mut camera = cameras[0]; // find(|c| c.info().contains("U3-368xXLE-NIR")).unwrap()
    camera.open().unwrap();
    camera.load_context().unwrap();          
    let nir = camera.start_streaming(3).unwrap();
 
    /*fn decode(path: impl AsRef<std::path::Path>) -> Image<Box<[u8]>> {
        let image = imagers::open(path).unwrap();
        Image::new(vector::xy{x: image.width(), y: image.height()}, image.into_bytes().into_boxed_slice())
    }
    let ir = decode("ir.png");*/

    struct View {
        nir: cameleon::payload::PayloadReceiver,
    }
    impl ui::Widget for View { 
        #[ui::throws(ui::Error)] fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) { 
            let payload = self.nir.recv_blocking().unwrap();
            //let Ok(payload) = self.camera.recv_blocking() else { return Ok(()) };
            //println!("blocking"); let Ok(payload) = payload_rx.recv_blocking() else { println!("continue"); continue; }; println!("ok");
            let &cameleon::payload::ImageInfo{width, height, ..} = payload.image_info().unwrap();
            fn neg(source: &[u8]) -> Box<[u8]> {
                let mut target = Box::new_uninit_slice(source.len());
                assert!(source.len()%64==0);
                unsafe {
                    use std::simd::u8x64;
                    let len = source.len();
                    let source = source.as_ptr() as *const u8x64;
                    {
                        let target = target.as_mut_ptr() as *mut u8x64;
                        for i in (0..len).step_by(64) {
                            target.byte_add(i).write(u8x64::splat(0xFF)-source.byte_add(i).read());
                        }
                    }
                    target.assume_init()
                }
            }
            let source = Image::new(xy{x: width as u32, y: height as u32}, neg(payload.image().unwrap()));
            downscale(target, source.as_ref());
            
            /*let nir = checkerboard(nir.as_ref());
            let mut P = checkerboards; P[1] = [P[1][0], P[1][3], P[1][1], P[1][2]];
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
