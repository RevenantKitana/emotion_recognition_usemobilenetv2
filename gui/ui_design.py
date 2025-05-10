import tkinter as tk

# Creating the main window
window = tk.Tk()
window.title("Emotion Recognition")

# Add labels and buttons to the interface
label = tk.Label(window, text="Emotion Recognition App")
label.pack()

button = tk.Button(window, text="Start", command=None)
button.pack()

# Start the GUI loop
window.mainloop()
