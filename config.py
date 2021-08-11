#  Writes out an example config file for this component's local and global
#  settings

import configparser


config = configparser.ConfigParser()

config.add_section("local")
config.set("local", "epsilon", ".99")
config.add_section("global")
config.set("global", "store_lstm_state", "True")

with open("settings.ini", "w") as configfile:  # save
    config.write(configfile)
