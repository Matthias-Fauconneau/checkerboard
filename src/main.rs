#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}};

pub trait Camera : Sized {
    fn new() -> Self;
    fn next(&mut self) -> Image<Box<[u16]>>;
    fn next_or_saved_or_start(camera: &mut Option<Self>, clone: &mut Option<Image<Box<[u16]>>>, name: &str, size: size) -> Image<Box<[u16]>> {
        if let Some(camera) = camera.as_mut() {let image = camera.next(); *clone = Some(image.clone()); image} 
        else if let Some(image) = raw(name,size) { image }
        else { *camera = Some(Self::new()); let image = camera.as_mut().unwrap().next(); *clone = Some(image.clone()); image}
    }
}
pub struct Unimplemented;
impl Camera for Unimplemented {
    fn new() -> Self { Self }
    fn next(&mut self) -> Image<Box<[u16]>> { unimplemented!() }
}

#[cfg(feature="nir")] mod nir; #[cfg(feature="nir")] use nir::NIR; #[cfg(not(feature="nir"))] type NIR = Unimplemented;
#[cfg(feature="ir")] mod ir; #[cfg(feature="ir")] use ir::IR; #[cfg(not(feature="ir"))] #[allow(unused)] type IR = Unimplemented;
#[cfg(feature="hololens")] mod hololens; #[cfg(feature="hololens")] use hololens::Hololens; #[cfg(not(feature="hololens"))] #[allow(unused)] type Hololens = Unimplemented;

mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;

struct App {
    nir: Option<NIR>,
    #[cfg(feature="ir")] ir: Option<IR>,
    #[cfg(feature="hololens")] hololens: Option<Hololens>,
    last_frame: [Option<Image<Box<[u16]>>>; 3],
    debug_which: &'static str,
    debug: &'static str,
}
impl App { fn new() -> Self { Self{#[cfg(feature="hololens")] hololens: None, nir: /*None*/Some(NIR::new()), #[cfg(feature="ir")] ir: None, last_frame: [None, None, None], debug_which: "nir", debug:"even"}}}
impl ui::Widget for App {
    fn size(&mut self, _: size) -> size { xy{x:2592,y:1944} }
    fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
        /*let hololens = Camera::next_or_saved_or_start(&mut self.hololens,&mut self.last_frame[0], "hololens",xy{x:1280,y:720});
        let debug = if self.debug_which=="hololens" {self.debug} else{""};
        if debug=="original" { scale(target, hololens.as_ref()); return Ok(()) }
        let points = match checkerboard_direct_intersections(hololens.as_ref(), 64, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(()) }
        let P_hololens = long_edge_first(top_left_first(simplify(convex_hull(&points.into_iter().map(vec2::from).collect::<Box<_>>())).try_into().unwrap()));
        if debug=="quads" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in P_hololens { cross(target, scale, offset, a, 0xFF00FF); } return Ok(()) }*/

        let nir = Camera::next_or_saved_or_start(&mut self.nir, &mut self.last_frame[1], "nir",xy{x:2592,y:1944});
        let nir = {let size = nir.size/128*128; nir.slice((nir.size-size)/2, size)};
        //let nir = {let x = nir.size.x/64*64; nir.slice(xy{x: (nir.size.x-x)/2, y: 0}, xy{x, y: nir.size.y})}; // Both dimensions need to be aligned because of transpose (FIXME: only align stride+height)
        let debug = if self.debug_which=="nir" {self.debug} else{""};
        if debug=="original" { scale(target, nir.as_ref()); return Ok(()) }
        let low = box_convolve::<4>(nir.as_ref());
        if debug=="low" { scale(target, low.as_ref()); return Ok(()) }
        let Some(high) = self::high_pass::<42>(low.as_ref(), 0x1000) else { scale(target, low.as_ref()); return Ok(()) };
        if debug=="even" { scale(target, high.as_ref()); return Ok(()) }        
        //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 9, 64., debug, target)  else { return Ok(()) };

        /*let ir = Camera::next_or_saved_or_start(&mut self.ir, &mut self.last_frame[2], "ir",xy{x:256,y:192});
        let debug = if self.debug_which=="ir" {self.debug} else{""};
        if debug=="original" { scale(target, ir.as_ref()); return Ok(()); }
        let Some(high) = self::high_pass(ir.as_ref(), 16, 0x2000) else { scale(target, ir.as_ref()); return Ok(()); };
        let points = match checkerboard_direct_intersections(high.as_ref(), 32, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, ir.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(())}
        let P_ir = fit_rectangle(&points);
        if debug=="quads" { let (_, scale, offset) = scale(target, ir.as_ref()); for a in P_ir { cross(target, scale, offset, a, 0xFF00FF); } return Ok(())}*/
        
        let map = |P:[[vec2; 4]; 2]| -> mat3 {
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            mul(inverse(M[1]), mul(A, M[0]))
        };
        
        /*let (target_size, scale, offset) = scale(target, hololens.as_ref());
        for (i,&p) in P_hololens.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
        affine_blit(target, target_size, nir.as_ref(), map([P_hololens, P_nir]), hololens.size, 2);
        affine_blit(target, target_size, ir.as_ref(), map([P_hololens, P_ir]), hololens.size, 1);*/

        let P_display = [xy{x: 0., y: 0.}, xy{x: target.size.x as f32, y: 0.}, xy{x: target.size.x as f32, y: target.size.y as f32}, xy{x: 0., y: target.size.y as f32}];
        let target_size = target.size;
        //affine_blit(target, target_size, nir.as_ref(), map([P_display, P_nir]), target.size, 1);
        Ok(())
    }
    fn event(&mut self, _: vector::size, context: &mut ui::EventContext, event: &ui::Event) -> ui::Result<bool> {
        if let &ui::Event::Key(key) = event {
            if ['⎙','\u{F70C}'].contains(&key) {
                println!("⎙");
                write("checkerboard.png", {let mut target = Image::uninitialized(xy{x: 2592, y:1944}); let size = target.size; self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap(); target}.as_ref());
                if self.last_frame.iter_mut().zip([/*"hololens",*/"nir"/*,"ir"*/]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| write_raw(name, image.as_ref())).count() == 0 {
                    //if self.hololens.is_none() { self.hololens = Some(Hololens::new()); }
                    if self.nir.is_none() { self.nir = Some(NIR::new()); }
                    //if self.ir.is_none() { self.ir = Some(IR::new()); }
                }
            } else {
                fn starts_with<'t>(words: &[&'t str], key: char) -> Option<&'t str> { words.into_iter().find(|word| key==word.chars().next().unwrap()).copied() }
                if let Some(word) = starts_with(&[/*"hololens",*/"nir"/*,"ir"*/], key) { self.debug_which = word }
                else if let Some(word) = starts_with(&["binary","contour","response","erode","?","low","max","original","z","points","quads","source"], key) { self.debug = word; }
                else { self.debug_which=""; self.debug="";  }
                context.set_title(&format!("{} {}", self.debug_which, self.debug));
            }
            Ok(true)
        } else { Ok(/*self.hololens.is_some() ||*/ self.nir.is_some() /*||self.ir.is_some()*/) }
    }
}
fn main() -> ui::Result { ui::run("Checkerboard", &mut App::new()) }
