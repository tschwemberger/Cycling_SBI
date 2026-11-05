import streamlit as st, torch, sys, tempfile, os
import numpy as np
from SBI_inference import ride_analysis, import_track
from SBI_embedding import PhysicsEmbedding
sys.modules['__main__'].PhysicsEmbedding = PhysicsEmbedding

st.title("Where are your watts going?")
use_prior = st.checkbox("Do you want to use your own estimate for the system mass? (If not, the model will infer it from the data.)", value=False)
if use_prior:
    sys_mass = st.slider("Approximate system mass (kg)", min_value=40, max_value=150, value=75, step=1)
else:
    sys_mass = None

up = st.file_uploader("GPX or FIT file with power", type=["gpx", "fit"])

if up is not None:
    ext = os.path.splitext(up.name)[1].lower()   # ".gpx" or ".fit"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(up.getbuffer())
        path = f.name
    st.write("File loaded.")
    st.write("Fetching weather data...")
    try:
        ride = import_track(path)
    finally:
        os.remove(path)

    with st.spinner("Inferring posterior distributions..."):
        params, (corner_fig, pie_fig), m0, work = ride_analysis(ride, sys_mass=sys_mass, NUM_SAMPLES=100000)
        work_inf, work_rec = work
    if use_prior:
        if np.abs(m0-sys_mass)/sys_mass > 0.15:
            st.warning(f"The estimated system mass of {m0:.1f} kg differs from input {sys_mass:.1f} kg by more than 15%. This may result in inaccurate estimates. If you're confident in your input mass, this may indicate a problem with the data such a power meter miscalibration or an inconsistency with the model such as unmodeled braking effects or strong correlations between parameters.")
    if np.abs(work_inf-work_rec)/work_rec > 0.1:
        st.warning(f'At the inferred paramters, the work required in the clean segments is {int(work_inf/1e3)} kJ while the recorded work is {int(work_rec/1e3)} kJ, a difference of {int(100*np.abs(work_inf-work_rec)/work_rec)}\%. This indicates the model may be a poor fit to the data.')

    st.write('## Distribution of Losses While Pedaling')
    st.write('The pie chart below shows the distribution of losses while pedaling, based on the inferred parameters. The model estimates how much of your power is going into overcoming gravity, aerodynamic drag, rolling resistance, and surface vibration losses. The remaining power drives your accelerations. This chart does not reflect the recovery you get while coasting.')
    st.pyplot(pie_fig)
    st.write('Rolling resistance is divided into two components, hysteretic losses $C_{rr}$ and surface impedance $f_{surf}$ with the latter spanning a wide range from smooth roads with $k_{surf}\lesssim 1$ to rough gravel, mud, or snow with $k_{surf}\sim 10$.')
    
    st.write('## Posterior Distributions of Parameters')
    st.pyplot(corner_fig)
    st.write('The corner plot above shows the posterior distributions of the parameters inferred from your ride data. The diagonal plots show the marginal distributions of each parameter, while the off-diagonal plots show the pairwise joint distributions. If the marginal distributions are piled at the end of the window, it may indicate that the true value is outside the range of the prior. A diagonal band in the joint distributions indicates a correlation between parameters.')
    
    st.write('The model assumes surface vibrational losses are a fixed function of only speed. If the quality of the surface is correlated with other parameters such as gratient, this may result in degeneracies in the inferred parameters. For instance, if rough surfaces are strongly correlated with steep gradients, the model will have difficulty distinguishing higher mass from higher surface losses. The resolution of the network is limited by the number of samples, GPS noise, and barometric glitches. The effect of hysteretical losses $(C_{rr})$ is generally below this threshold, so that panel should be viewed as a rough estimate, largely degenerate with surface losses $(k_{surf})$.')
    
    st.write('Also note that the function defining surface losses is not well studied. As such, it is defined with a wide range of marginalized parameters allowing it to act as a "sink" term, soaking up all losses which do not fit the other, well modeled parameters. This makes it particularly sensitive to features like unmodeled braking.')