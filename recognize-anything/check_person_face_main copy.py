# command ex) python inference_ram_plus_webcam2.py --pretrained pretrained/ram_plus_swin_large_14m.pth --interval 20 

import def_person
import def_face
def main():
   
   print ("!!!!!!! person chk start")

   def_person.main()
      
   print ("!!!!!!! person chk ended")


   print ("!!!!!!! face chk start")

   def_face.main()
      
   print ("!!!!!!! face chk ended")



if __name__ == "__main__":
    main()