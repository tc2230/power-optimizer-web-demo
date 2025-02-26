# import streamlit as st


# if "a" not in st.session_state:
#     st.session_state["a"] = {}
#     st.session_state["a"]["label"] = "some text"
#     st.session_state["a"]["value"] = 123
#     st.session_state["a"]["help"] = "some text"

# a = st.number_input(
#     label = "label",
#     value = 123,
#     help = "help",
#     key = "a",
#     )

# st.write(st.session_state["a"])
# setattr(st.session_state, "btn_opt_clicked", True)
# st.write(st.session_state["btn_opt_clicked"])

# st.write(":warning:")
# st.session_state["val"] = 1

import streamlit as st

name = st.text_input("Name")
if not name:
  st.warning('Please input a name.')
  st.stop()

st.success("Thank you for inputting a name.")
