from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

root = Tk()
root.title('Traffic Number Plate Reader')
root.iconbitmap('c:/Users/Debjit/OneDrive/Pictures/Camera Roll')

root.filename = filedialog.askopenfilename(initialdir = '/Users/Debjit/OneDrive/Pictures/Camera Roll', title = "Select a file", filetypes = (("png files", "*.png"),("jpg files", "*.jpg")))
my_btn1 = Button(root, text = "Submit", command = open)

myLabel = Label(root, text = "Select the language in which the number plate is recorded")
myLabel.pack()

clicked = StringVar()
clicked.set("Bengali")

drop = OptionMenu(root, clicked, "Bengali", "Hindi", "Tamil", "Kannada", "Telugu")
myButton = Button(root, text = " ", command = show).pack()


root.mainloop()