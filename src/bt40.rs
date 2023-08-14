#!/bin/sh
#![allow(dead_code)] /*
modprobe usb-serial-simple
echo 04b8 0d12 >/sys/bus/usb-serial/drivers/generic/new_id
chmod a+rw /dev/ttyUSB0
exit #*/
pub struct BT40(Box<dyn serialport::SerialPort>);
impl BT40 {
    pub fn new() -> Self { Self(serialport::new("/dev/ttyUSB0", 115200).timeout(std::time::Duration::from_millis(1000)).open().unwrap()) }
    fn command(&mut self, command: impl AsRef<str>) -> String {
        let command = command.as_ref();
        println!("{command}");
        self.0.write(format!("{command}\r\n").as_bytes()).unwrap(); 
        let mut read = |len| {let mut buffer: Vec<u8> = vec![0; len]; let len = self.0.read(&mut buffer).unwrap(); buffer.truncate(len); String::from_utf8(buffer).unwrap()};
        loop {
            let r = read(command.len());
            if r == command { break; }
            if !r.trim().is_empty() && r.trim()!=":" { use std::io::Write; std::io::stdout().write_all(r.as_bytes()).unwrap(); }
        }
        let mut r = read(8);
        assert_eq!(&r[0..2], "\r\n"); r.drain(0..2);
        while ["\r","\n"].contains(&&r[r.len()-1..]) { r.pop(); }
        print!("'{r}'");
        r
    }
    pub fn enable(&mut self) { assert_eq!(self.command("setmute 0"), "OK"); }
    pub fn disable(&mut self) { assert_eq!(self.command("setmute 1"), "OK"); }
}
impl Drop for BT40 { fn drop(&mut self) { self.disable() }}