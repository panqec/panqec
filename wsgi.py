from panqec.gui import GUI

gui = GUI()
app = gui.app

if __name__ == "__main__":
    gui.run(host='0.0.0.0', debug=False, port=5000)
