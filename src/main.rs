#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;
mod hololens; use hololens::*;
mod nir; use nir::*;
mod ir; use ir::*;

fn main() {
    let hololens = /*std::env::args().any(|a| a=="hololens").*/true.then(Hololens::new);
    let nir = std::env::args().any(|a| a=="nir").then(NIR::new);    
    let ir = std::env::args().any(|a| a=="ir").then(IR::new);

    struct View {
        hololens: Option<Hololens>,
        nir: Option<NIR>,
        ir: Option<IR>,
        last_frame: [Option<Image<Box<[u16]>>>; 3],
        debug: &'static str,
        debug_which: &'static str,
    }
    impl ui::Widget for View {
        fn size(&mut self, _: size) -> size { xy{x: 2592, y: 1944} }
        fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
            let hololens = self.hololens.as_mut().map(|hololens| {let hololens = hololens.next(); self.last_frame[0] = Some(hololens.clone()); hololens}).unwrap_or_else(||open("hololens.png"));
            if self.debug_which=="hololens" && ["original","source"].contains(&self.debug) { scale(target, hololens.as_ref()); return Ok(()) }
            let Some(P_hololens) = checkerboard_quad_debug(hololens.as_ref(), true, if self.debug_which=="hololens" {self.debug} else{""}, target)  else { return Ok(()) };

            let nir = self.nir.as_mut().map(|nir|{let nir = nir.next(); self.last_frame[1] = Some(nir.clone()); nir}).unwrap_or_else(||open("nir.png"));
            if self.debug_which=="nir" && ["original","source"].contains(&self.debug) { scale(target, nir.as_ref()); return Ok(()) }
            let Some(P_nir) = checkerboard_quad_debug(nir.as_ref(), true, if self.debug_which=="nir" {self.debug} else{""}, target)  else { return Ok(()) };

            let ir = self.ir.as_mut().map(|ir|{let ir = ir.next(); self.last_frame[2] = Some(ir.clone()); ir}).unwrap_or_else(||raw(xy{x:256,y:192}, "ir"));
            if self.debug_which=="ir" && ["original","source"].contains(&self.debug) { scale(target, ir.as_ref()); return Ok(()); }
            let points = match checkerboard_direct_intersections(ir.as_ref(), 32, if self.debug_which=="ir" {self.debug} else{""}) {
                Ok(points) => points,
                Err(image) => {scale(target, image.as_ref()); return Ok(()) }
            };
            if self.debug_which=="ir" && self.debug=="points" {
                let (_, scale, offset) = scale(target, ir.as_ref());
                for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); }
                return Ok(())
            }
            let area = |[a,b,c,d]:[vec2; 4]| {let abc = vector::cross2(b-a,c-a); let cda = vector::cross2(d-c,a-c); (abc+cda)/2.};
            use std::f32::consts::PI;
            let P_ir = [0./*,PI/2.*/].map(|_angle| {
                // TODO: rotate
                let vector::MinMax{min, max} = vector::minmax(points.iter().map(|p| p.map(|u32| u32 as f32))).unwrap();    
                [xy{x: min.x, y: min.y}, xy{x: max.x, y: min.y}, xy{x: max.x, y: max.y}, xy{x: min.x, y: max.y}]
                // TODO: rotate back
            }).into_iter().min_by(|&a,&b| f32::total_cmp(&area(a), &area(b))).unwrap();           
            if self.debug_which=="ir" && self.debug=="quads" {
                let (_, scale, offset) = scale(target, ir.as_ref());
                for a in P_ir { cross(target, scale, offset, a, 0xFF00FF); }
                return Ok(())
            }
            
            let P = [P_hololens, P_nir];
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            let A = mul(inverse(M[1]), mul(A, M[0]));

            let (target_size, scale, offset) = scale(target, hololens.as_ref());
            for (i,&p) in P_hololens.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
            affine_blit(target, target_size, nir.as_ref(), A, hololens.size);
            Ok(())
        }
        fn event(&mut self, _: vector::size, _: &mut Option<ui::EventContext>, event: &ui::Event) -> ui::Result<bool> {
            use ui::Event::Key;
            match event {
                Key('a') =>{ self.debug = ""; return Ok(true); }
                Key('b') =>{ self.debug = "binary"; return Ok(true); }
                Key('c') if self.debug_which=="nir" =>{ self.debug = "contour"; return Ok(true); }
                Key('c') if self.debug_which=="ir" =>{ self.debug = "cross"; return Ok(true); }
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
                //Key('p') =>{ self.debug = "peaks"; return Ok(true); }
                Key('p') =>{ self.debug = "points"; return Ok(true); }
                Key('q') =>{ self.debug = "quads"; return Ok(true); }
                //Key('s') =>{ self.debug = "selection"; return Ok(true); }
                Key('s') =>{ self.debug = "source"; return Ok(true); }
                Key('⎙')|Key('\u{F70C}')/*|Key(' ')*/ => {
                    println!("⎙");
                    if self.last_frame.iter_mut().zip(["hololens","nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| std::fs::write(name, &bytemuck::cast_slice(image)).unwrap()).count() == 0 {
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
    ui::run("Checkerboard", &mut View{hololens, nir, ir, last_frame: [None, None], debug:"original", debug_which: "hololens"}).unwrap();
}
