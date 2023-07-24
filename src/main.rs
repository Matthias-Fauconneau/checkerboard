#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}};
mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;

pub trait Camera : Sized {
    fn new() -> Self;
    fn next(&mut self) -> Image<Box<[u16]>>;
    fn next_or_saved_or_start(camera: &mut Option<Self>, clone: &mut Option<Image<Box<[u16]>>>, name: &str, size: size) -> Image<Box<[u16]>> {
        if let Some(camera) = camera.as_mut() {let image = camera.next(); *clone = Some(image.clone()); image} 
        else if let Some(image) = raw(name,size) { image }
        else { *camera = Some(Self::new()); let image = camera.as_mut().unwrap().next(); *clone = Some(image.clone()); image}
    }
}
mod hololens; use hololens::*;
mod nir; use nir::*;
mod ir; use ir::*;

fn main() {
    let hololens = std::env::args().any(|a| a=="hololens").then(Hololens::new);
    let nir = std::env::args().any(|a| a=="nir").then(NIR::new);    
    let ir = std::env::args().any(|a| a=="ir").then(IR::new);

    struct View {
        hololens: Option<Hololens>,
        nir: Option<NIR>,
        ir: Option<IR>,
        last_frame: [Option<Image<Box<[u16]>>>; 3],
        debug_which: &'static str,
        debug: &'static str,
    }
    impl ui::Widget for View {
        fn size(&mut self, _: size) -> size { xy{x:1280,y:720} }
        fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
            
            let hololens = Camera::next_or_saved_or_start(&mut self.hololens,&mut self.last_frame[0], "hololens",xy{x:1280,y:720});
            let debug = if self.debug_which=="hololens" {self.debug} else{""};
            if debug=="original" { scale(target, hololens.as_ref()); return Ok(()) }
            //let Some(P_hololens) = checkerboard_quad_debug(hololens.as_ref(), true, 3, 20., if self.debug_which=="hololens" {self.debug} else{""}, target)  else { return Ok(()) };
            let points = match checkerboard_direct_intersections(hololens.as_ref(), 64, debug) {
                Ok(points) => points,
                Err(image) => {scale(target, image.as_ref()); return Ok(()) }
            };
            if debug=="points" {
                let (_, scale, offset) = scale(target, hololens.as_ref());
                for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); }
                return Ok(())
            }
            let P_hololens = long_edge_first(top_left_first(simplify(convex_hull(&points.into_iter().map(vec2::from).collect::<Box<_>>())).try_into().unwrap()));
            if debug=="quads" {
                let (_, scale, offset) = scale(target, hololens.as_ref());
                for a in P_hololens { cross(target, scale, offset, a, 0xFF00FF); }
                return Ok(())
            }

            let nir = Camera::next_or_saved_or_start(&mut self.nir, &mut self.last_frame[1], "nir",xy{x:2592,y:1944});
            let debug = if self.debug_which=="nir" {self.debug} else{""};
            if debug=="original" { scale(target, nir.as_ref()); return Ok(()) }
            let low = low_pass(nir.as_ref(), 4/*12*/);
            
            if debug=="low" { scale(target, low.as_ref()); return Ok(()) }
            let Some(high) = self::high_pass(low.as_ref(), 42/*127*/, 0x1000) else { scale(target, low.as_ref()); return Ok(()) };
            if debug=="high" { scale(target, high.as_ref()); return Ok(()) }        
            let Some(P_nir) = checkerboard_quad_debug(nir.as_ref(), true, 9, 64., debug, target)  else { return Ok(()) };

            let ir = Camera::next_or_saved_or_start(&mut self.ir, &mut self.last_frame[2], "ir",xy{x:256,y:192});
            let debug = if self.debug_which=="ir" {self.debug} else{""};
            if debug=="original" { scale(target, ir.as_ref()); return Ok(()); }
            let Some(high) = self::high_pass(ir.as_ref(), 16, 0x2000) else { scale(target, ir.as_ref()); return Ok(()); };
            let points = match checkerboard_direct_intersections(high.as_ref(), 32, debug) {
                Ok(points) => points,
                Err(image) => {scale(target, image.as_ref()); return Ok(()) }
            };
            if debug=="points" {
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
            if debug=="quads" {
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
            if let &ui::Event::Key(key) = event {
                if ['⎙','\u{F70C}'].contains(&key) {
                    println!("⎙");
                    if self.last_frame.iter_mut().zip(["hololens","nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| write_raw(name, image.as_ref())).count() == 0 { self.ir = Some(IR::new()); }
                    write("checkerboard.png", {let mut target = Image::uninitialized(xy{x: 2592, y:1944}); let size = target.size; self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap(); target}.as_ref());
                }
                else if let Some(word) = ["hololens","nir","ir"].iter().find(|word| key==word.chars().next().unwrap()) { self.debug_which = word }
                else if let Some(word) = ["binary","contour","response","erode","high","low","max","original","checkerboard","points","quads","source"].iter().find(|word| key==word.chars().next().unwrap()) { self.debug = word; }
                else { self.debug=""; self.debug_which=""; }
                Ok(true)
            } else { Ok(self.hololens.is_some() || self.nir.is_some()||self.ir.is_some()) }
        }
    }
    ui::run("Checkerboard", &mut View{hololens, nir, ir, last_frame: [None, None, None], debug_which: "hololens", debug:"points"}).unwrap();
}
