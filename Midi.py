import rtmidi
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF

midiout = rtmidi.MidiOut()
midiin = rtmidi.MidiIn() 
midi_channel = 1            # 1-16
available_programs = ["Acoustic", "Stratocaster", "Telecaster"]
program_index_required = 0   # start from 0

required_device = "Bela"        #0 = virtual, 1 = bela, 2 = audiobox

""" virtual port name = 'IAC Driver Virtual Port'
    bela port name = 'Bela'
    
"""

def get_midi_ports(required_device):
    
    port_count = midiout.get_port_count()
    print(str(port_count) + " MIDI ports available:")
    
    for i in range(port_count):
        print("                       Index " + str(i) + ": '" + str(midiout.get_port_name(i)) +"'")
        if (midiout.get_port_name(i) == required_device):
            required_device_index = i
    print("\n")
    
    return required_device_index
    
       
def virtual_port():
    
    port_count = midiout.get_port_count()
    print(str(port_count) + " MIDI ports available:")
    
    for i in range(port_count):
        print("                       Index " + str(i) + ": '" + str(midiout.get_port_name(i)) +"'")
    midiout.close_port()

def is_out_port_open():
    port_status = midiout.is_port_open()
    if (port_status == False):
        print("Port is closed")
    elif (port_status == True):
        print("Port is open")
        
def print_midi_bytes():
   
    print("Program change status bytes available:")
    for i in range(16):
        decimal = 192 + i
        hexidecimal_value = hex(decimal)
        binary_value = bin(decimal).replace('0b','')
        binary_value = int(binary_value)
        binary_num = f"{binary_value:08}"
        if (i < 9):
            print("Midi Channel " + str(i+1) + "   | Hex: " + hexidecimal_value + "    | Binary: " + binary_num + "    | Decimal: " + str(decimal))
        else:
            print("Midi Channel " + str(i+1) + "  | Hex: " + hexidecimal_value + "    | Binary: " + binary_num+ "    | Decimal: " + str(decimal))
            
    print("\nProgram change data bytes available:")
    program_index = 0   # initialise to 0 prior to iterating through available programs
    for value in available_programs:
        decimal = program_index
        hexidecimal_value = hex(decimal)
        binary_value = bin(decimal).replace('0b','')
        binary_value = int(binary_value)
        binary_num = f"{binary_value:08}"
        print("Program name: " + value + "   | Hex: " + hexidecimal_value + "    | Binary: " + binary_num + "    | Decimal: " + str(decimal))
        program_index+=1
        
        
def midi_message_info(midi_channel, program_num):
    program_binary_num = program_num
    print("\nSelected Status Byte:")
    channel_decimal = 191 + midi_channel    # first int value is 192, so for midi channel 1 - 192 + 1 = 192
    channel_hexidecimal = hex(channel_decimal)
    channel_binary = bin(channel_decimal).replace('0b','')
    channel_binary = int(channel_binary)   # cast back to int in order to extend binary to 8 digits for full midi message
    channel_binary_num = f"{channel_binary:08}"
    print("Midi Channel " + str(midi_channel) + "   | Hex: " + channel_hexidecimal + "    | Binary: " + channel_binary_num + "    | Decimal: " + str(channel_decimal))
   
    print("Selected Data Byte:")
    program_binary_num = program_num
    program_decimal = program_num
    program_hexidecimal = hex(int(program_decimal))
    program_binary = bin(int(program_decimal)).replace('0b','')
    program_binary = int(program_binary)
    program_binary_num = f"{program_binary:08}"
    print("Program name: " + available_programs[int(program_decimal)] + "   | Hex: " + program_hexidecimal + "    | Binary: " + program_binary_num + "    | Decimal: " + str(program_decimal))
    
    return channel_hexidecimal

    
        
def send_midi_message(midi_port_index, channel_num, program_num):
    port = midiout.get_port_name(midi_port_index)
    midiout.open_port(midi_port_index)
    if (midiout.is_port_open() == True):
        print("\n'" + str(port) + "' port has opened")
        channel_num = int(channel_num,0)
        program_change_message = [channel_num, program_num] 
        midiout.send_message(program_change_message)
        #midiout.send_message([NOTE_ON, 70, 100])
        print("message = " + str(program_change_message))
        midiout.close_port()
        print("'" + str(port) + "' port has closed")
    elif (midiout.is_port_open() == False):
        print("Error: could not open port named '" + str(port) + "'")

        
if __name__ == "__main__":
    required_device_index = get_midi_ports(required_device= required_device)
    print_midi_bytes()
    channel_hex = midi_message_info(midi_channel=midi_channel, program_num=program_index_required)
    send_midi_message(midi_port_index=required_device_index, channel_num=channel_hex, program_num=program_index_required)
   
    #is_out_port_open()
