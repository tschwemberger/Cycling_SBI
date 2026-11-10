import streamlit as st, torch, sys, tempfile, os
import numpy as np
import xml.etree.ElementTree as ET
import SBI_inference
from SBI_inference import ride_analysis, import_track, plot_fracs_grad
import matplotlib.pyplot as plt
from SBI_embedding import PhysicsEmbedding
import fitdecode
import hashlib
sys.modules['__main__'].PhysicsEmbedding = PhysicsEmbedding

@st.fragment
def power_plot_section(params, rho0, default_power):
    st.write('### Distribution of losses as a function of gradient')
    st.write('Here you can see how much each force costs you at a steady speed and gradient assuming no wind.')
    power_plot = st.slider("Target power (W)", 50, 500, default_power, 1)
    grad_fig, _ = plot_fracs_grad(params, power_plot, rho0)
    st.pyplot(grad_fig)
    plt.close(grad_fig)


st.title("Where are your watts going?")

up = st.file_uploader("GPX or FIT file with power", type=["gpx", "fit"])

use_prior = st.checkbox("Do you want to use your own estimate for the system mass? (If not, the model will infer it from the data.)", value=False)
if use_prior:
    sys_mass = st.slider("Approximate system mass (kg)", min_value=40, max_value=150, value=75, step=1)
else:
    sys_mass = None

if up is not None:
    file_hash = hashlib.md5(up.getvalue()).hexdigest()
    # Level 1: parsing + weather depend only on the file
    if st.session_state.get("ride_key") != file_hash:
        ext = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(up.getbuffer())
            path = f.name
        st.write("File loaded.")
        with st.spinner("Fetching weather data..."):
            try:
                st.session_state.ride = import_track(path)
            except ET.ParseError as e:
                ...  # your existing error handling, unchanged
            finally:
                os.remove(path)
        st.session_state.ride_key = file_hash
    ride = st.session_state.ride

    # Level 2: inference depends on the file and the mass prior
    analysis_key = (file_hash, sys_mass)
    if st.session_state.get("analysis_key") != analysis_key:
        with st.spinner("Inferring posterior distributions..."):
            st.session_state.analysis = ride_analysis(ride, sys_mass=sys_mass, NUM_SAMPLES=100000)
        st.session_state.analysis_key = analysis_key
    params, (corner_fig, pie_fig, hist_fig), (m_lo, m0, m_hi), work, num_segs, test = st.session_state.analysis
    work_inf, work_rec = work

    ext = os.path.splitext(up.name)[1].lower()   # ".gpx" or ".fit"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(up.getbuffer())
        path = f.name
    
    t, d, x, y, z, z_gps, z_bar, grad, power, speed, acceleration, wind, rho = ride
    st.write(f'{num_segs} clean 90s windows identified')
    if num_segs < 30:
        st.warning("This may not be enough data to make reasonable inferences")

    if use_prior:
        if (sys_mass < m_lo) or (sys_mass > m_hi):
            st.warning(f"The network infers a system mass from {m_lo:.1f} to {m_hi:.1f} with 68% confidence. This disagrees with your specified mass and may indicate a problem with the data such a power meter miscalibration or an inconsistency with the model such as unmodeled braking effects or strong correlations between parameters.")

    if np.abs(work_inf-work_rec)/work_rec > 0.1:
        st.warning(f'At the inferred paramters, the work required in the clean segments is {int(work_inf/1e3)} kJ while the recorded work is {int(work_rec/1e3)} kJ, a difference of {int(100*np.abs(work_inf-work_rec)/work_rec)}%. This indicates the model may be a poor fit to the data.')

    st.write('## Distribution of Losses While Pedaling')
    st.write('The pie chart below shows the distribution of losses while pedaling, based on the inferred parameters. The model estimates how much of your power is going into overcoming gravity, aerodynamic drag, rolling resistance, and surface vibration losses. The remaining power drives your accelerations. This chart does not reflect the recovery you get while coasting.')
    st.pyplot(pie_fig)
    st.write('Rolling resistance is divided into two components, hysteretic losses $C_{rr}^{(h)}$ and surface impedance, parameterized by $C_{rr}^{(i)}$ with the latter spanning a wide range from smooth roads with $C_{rr}^{(i)}\lesssim 1$ to rough gravel, mud, or snow with $C_{rr}^{(i)}\sim 10$. I will provide additional details of this surface resistance function is forthcoming documentation.')

    power_plot_section(params, rho[0], int(np.mean(power)))
    
    st.write('## Posterior Distributions of Parameters')
    st.pyplot(corner_fig)
    st.write('The corner plot above shows the posterior distributions of the parameters inferred from your ride data. The diagonal plots show the marginal distributions of each parameter, while the off-diagonal plots show the pairwise joint distributions. If the marginal distributions are piled at the end of the window, it may indicate that the true value is outside the range of the prior. A diagonal band in the joint distributions indicates a correlation between parameters.')
    
    st.write('The model assumes surface vibrational losses are a fixed function of only speed. If the quality of the surface is correlated with other parameters such as gradient, this may result in degeneracies in the inferred parameters. For instance, if rough surfaces are strongly correlated with steep gradients, the model will have difficulty distinguishing higher mass from higher surface losses. The resolution of the network is limited by the number of samples, GPS noise, and barometric glitches. The effect of hysteretical losses $(C_{rr})$ is generally below this threshold, so that panel should be viewed as a rough estimate, largely degenerate with surface losses $(C_{rr}^{(i)})$.')
    
    st.write('Also note that the function defining surface losses is not well studied. As such, it is defined with a wide range of marginalized parameters allowing it to act as a "sink" term, soaking up all losses which do not fit the other, well modeled parameters. This makes it particularly sensitive to features like unmodeled braking.')

    show_hist = st.checkbox("Show selection bias in segment filering", value=False)
    if show_hist:
        st.write('### Selection of Data')
        st.write('The selection criteria for the 90s windows may bias your results towards climbing sections since they are much more likely to survive the filters. Here you can see how the full data set compares to the subset which the network finds to be useable. Note that all steep descents are removed as are corners, suspected braking, and hike-a-bikes.')
        st.pyplot(hist_fig)