#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows,slice_take,stdsimd)]
#![allow(non_camel_case_types,non_snake_case)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}, ui::time};

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
    last: Option<[vec2; 4]>,
    calibration: Option<[vec2; 4]>,
}
impl App { fn new() -> Self { Self{#[cfg(feature="hololens")] hololens: None, nir: /*None*/Some(NIR::new()), #[cfg(feature="ir")] ir: None, last_frame: [None, None, None], debug_which: "visible", debug:"", last: None, calibration: None}}}
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
        let debug = if self.debug_which=="visible" {self.debug} else{""};
        if debug=="original" { scale(target, nir.as_ref()); return Ok(()) }
        let low = blur::<4>(nir.as_ref()); // 9ms
        if debug=="blur" { scale(target, low.as_ref()); return Ok(()) }
        /*let high = self::normalize::<42>(low.as_ref(), 0x1000);
        if debug=="normalize" { scale(target, high.as_ref()); return Ok(()) }*/
        let high = low.as_ref();
        //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 0/*Blur is enough*//*9*/, /*min_side: */64., debug, target)  else { return Ok(()) };*/
        //let high = nir.as_ref();
        let points = match checkerboard_direct_intersections(high.as_ref(), /*max_distance:*/64, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, high.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(()) }
        let P_nir = long_edge_first(top_left_first(simplify(convex_hull(&points.into_iter().map(vec2::from).collect::<Box<_>>())).try_into().unwrap()));
        self.last = Some(P_nir);
        if debug=="quads" { let (_, scale, offset) = scale(target, nir.as_ref()); for a in P_nir { cross(target, scale, offset, a, 0xFF00FF); } return Ok(()) }


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

        let P_nir_identity = [xy{x: 0., y: 0.}, xy{x: nir.size.x as f32, y: 0.}, xy{x: nir.size.x as f32, y: nir.size.y as f32}, xy{x: 0., y: nir.size.y as f32}];
        if let Some(calibration) = self.last {
            let (_target_size, scale, offset) = scale(target, nir.as_ref());
            let map = map([P_nir_identity, calibration]);
            let P = P_nir.map(|p| apply(map, p));
            for (i,&p) in P.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }
            let [p00,p10,p11,p01] = P;
            for i in 0..5 { for j in 0..7 {
                let color = if (i%2==1) ^ (j%2==1) { 0xFFFFFF } else { 0 };
                let p = |di,dj| {
                    let [i, j] = [i+di, j+dj];
                    let [u, v] = [i as f32/5., j as f32 / 7.];
                    let p0 = (1.-u)*p00+u*p01;
                    let p1 = (1.-u)*p10+u*p11;
                    let p = (1.-v)*p0+v*p1;
                    vec2::from(offset)+scale*p
                };
                let P : [vec2; 4] = [p(0,0),p(0,1),p(1,1),p(1,0)];
                let vector::MinMax{min, max} = vector::minmax(P).unwrap();
                let [p00,p10,p11,p01] = P;
                for y in min.y as u32..=max.y as u32 { for x in min.x as u32..=max.x as u32 {
                    let p = vec2::from(xy{x,y});
                    if [p00,p10,p11,p01,p00].array_windows().all(|&[a,b]| vector::cross2(p-a,b-a)<0.) { target[xy{x,y}] = color; }
                }}
            }}
        } else {
            let target_size = target.size;
            affine_blit(target, target_size, nir.as_ref(), map([P_nir_identity, P_nir]), target.size, 1);
        }
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
            } else if key=='\n' {
                self.calibration = self.last;
                context.set_title(&format!("{}", self.calibration.is_some()));
            } else {
                fn starts_with<'t>(words: &[&'t str], key: char) -> Option<&'t str> { words.into_iter().find(|word| key==word.chars().next().unwrap()).copied() }
                if let Some(word) = starts_with(&[/*"hololens",*/"visible"/*,"ir"*/], key) { self.debug_which = word }
                else if let Some(word) = starts_with(&["blur","contour","response","erode","normalize","low","max","original","z","points","quads","threshold"], key) { self.debug = word; }
                else { self.debug_which=""; self.debug="";  }
                context.set_title(&format!("{} {}", self.debug_which, self.debug));
            }
            Ok(true)
        } else { Ok(/*self.hololens.is_some() ||*/ self.nir.is_some() /*||self.ir.is_some()*/) }
    }
}
#[cfg(debug_assertions)] const TITLE: &'static str = "#[cfg(debug_assertions)]";
#[cfg(not(debug_assertions))] const TITLE: &'static str = "Checkerboard";
fn main() -> ui::Result { ui::run(TITLE, &mut App::new()) }
