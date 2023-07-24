use image::{Image, xy};
pub struct Hololens {stream: ffmpeg::format::context::Input, video_stream_index: usize, decoder: ffmpeg::codec::decoder::Video}
impl super::Camera for Hololens {
    fn new() -> Self {
        use ffmpeg::{format, media::Type, codec::context::Context as CodecContext};
        ffmpeg::init().unwrap();
        let hololens = std::fs::read_to_string("../../../.hololens").unwrap();
        let hololens = hololens.trim();
        let stream = format::input(&format!("https://{hololens}@192.168.0.102/api/holographic/stream/live_high.mp4?holo=false&pv=true&mic=false&loopback=false&RenderFromCamera=false")).unwrap();
        let video = stream.streams().best(Type::Video).unwrap();
        let video_stream_index = video.index();
        let codec = CodecContext::from_parameters(video.parameters()).unwrap();
        let decoder = codec.decoder().video().unwrap();
        Self{stream, video_stream_index, decoder}
    }
    fn next(&mut self) -> Image<Box<[u16]>> {
        let mut frame = ffmpeg::util::frame::video::Video::empty();
        while self.decoder.receive_frame(&mut frame).is_err() {
            let (_, packet) = self.stream.packets().find(|(stream,_)| stream.index() == self.video_stream_index).unwrap();
            self.decoder.send_packet(&packet).unwrap();
        }
        Image::from_iter(xy{x: frame.width(), y: frame.height()}, frame.data(0).iter().map(|&u8| u8 as u16))
    }
}
