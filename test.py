import streamlit as st


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

def foo():
    st.write('exec 1')
    st.write(list(st.session_state.keys()))
    if 'test' not in st.session_state:
        st.session_state['test'] = 123

st.write('exec 2')
st.write(list(st.session_state.keys()))

form = st.form(key='form', clear_on_submit=False)
btn = form.form_submit_button('btn', on_click=foo)

st.write('exec 3')
st.write(list(st.session_state.keys()))
