import math
from typing import List

import altair as alt
import pandas as pd
import streamlit as st


def simulate_trajectory(
    initial_height: float,
    m: float,
    u: float,
    surface_pressure_mb: float,
    surface_temp_celsius: float,
    v0: float,
    CdA: float,
    dt: float = 0.001,
):
    """Simulate the trajectory of the falling UA using a kinematic model."""
    # Constants
    g = 9.81
    R = 287.05
    L = 0.0065

    surface_temp_kelvin = surface_temp_celsius + 273.15
    surface_pressure = surface_pressure_mb * 100

    current_temp_kelvin = surface_temp_kelvin - L * initial_height
    current_pressure = surface_pressure * (1 - (L * initial_height) / surface_temp_kelvin) ** (g / (R * L))
    initial_air_density = current_pressure / (R * current_temp_kelvin)

    # Initial conditions
    x = 0.0
    z = initial_height
    vx = 0.0
    vz = 0.0
    x_wind = 0.0
    vxx = v0
    time = 0.0

    times: List[float] = [time]
    z_values: List[float] = [z]
    x_values: List[float] = [x]

    air_density_values = [initial_air_density]
    temperature_values = [current_temp_kelvin]

    while z > 0:
        current_temp_kelvin = surface_temp_kelvin - L * z
        temperature_values.append(current_temp_kelvin)

        current_pressure = surface_pressure * (1 - (L * z) / surface_temp_kelvin) ** (g / (R * L))
        air_density = current_pressure / (R * current_temp_kelvin)
        air_density_values.append(air_density)

        u_term = math.sqrt((2 * m * g) / (air_density * CdA)) if air_density * CdA != 0 else math.inf

        F = 0.5 * air_density * (u - vx) ** 2 * CdA
        k = (air_density * CdA) / (2 * m)

        v = math.sqrt(vx**2 + vz**2)
        v2 = math.sqrt(vxx**2 + vz**2)

        ax = F / m
        az = -g - (k * v * vz)
        bx = -(k * v2 * vxx)

        vx += ax * dt
        vz += az * dt
        vxx += bx * dt

        vx = min(vx, u)
        if vz > u_term:
            vz = u_term

        x += (vx + vxx) * dt
        z += vz * dt
        x_wind += vx * dt
        time += dt

        times.append(time)
        z_values.append(z)
        x_values.append(x)

    final_vx = vx
    final_vxx = vxx
    final_vz = vz
    final_vxvz = math.sqrt(final_vx**2 + final_vz**2)
    final_x = x
    xvx = x - x_wind

    grz_radius = abs(x_wind)
    grz_area = grz_radius**2 * math.pi
    grz_min = xvx - x_wind
    p_coll = (1 / grz_area) if grz_area != 0 else math.inf

    horizontal_total = final_vx + final_vxx
    if final_vz == 0 and horizontal_total == 0:
        angle_from_vertical_deg = 0.0
    else:
        angle_from_vertical_deg = math.degrees(math.atan2(abs(horizontal_total), abs(final_vz)))

    trajectory_df = pd.DataFrame(
        {
            "Time (s)": times,
            "Altitude (m)": z_values,
            "Horizontal Distance (m)": x_values,
        }
    )

    return {
        "initial_temperature": temperature_values[0],
        "initial_air_density": air_density_values[0],
        "final_x": final_x,
        "xvx": xvx,
        "final_x_wind": x_wind,
        "final_vx": final_vx,
        "final_vxx": final_vxx,
        "final_vz": final_vz,
        "final_vxvz": final_vxvz,
        "total_time": time,
        "grz_origin": xvx,
        "grz_area": grz_area,
        "grz_radius": grz_radius,
        "grz_max": final_x,
        "grz_min": grz_min,
        "p_coll": p_coll,
        "angle_from_vertical_deg": angle_from_vertical_deg,
        "trajectory": trajectory_df,
    }


def main():
    st.set_page_config(page_title="UA Drop Calculator", layout="wide")
    st.title("Uncrewed Aircraft Drop Calculator")
    st.markdown(
        """
        This app simulates the trajectory of a falling uncrewed aircraft (UA) while accounting for wind effects. \
        Provide the same inputs as the original calculator and press **Run Simulation** to see the results.
        """
    )

    with st.sidebar:
        st.header("Simulation Inputs")
        initial_height = st.number_input("Initial height (meters)", min_value=0.0, value=50.0, step=1.0)
        m = st.number_input("Mass (kg)", min_value=0.0, value=5.0, step=0.1)
        u = st.number_input("Wind speed (m/s)", value=5.0, step=0.1)
        surface_pressure_mb = st.number_input("Surface pressure (mb)", min_value=0.0, value=1013.25, step=1.0)
        surface_temp_celsius = st.number_input("Surface temperature (°C)", value=15.0, step=0.1)
        v0 = st.number_input("Initial velocity (m/s)", value=0.0, step=0.1)
        CdA = st.number_input("CdA value", min_value=0.0, value=0.116, step=0.001, format="%.3f")

        run_simulation = st.button("Run Simulation")

    if run_simulation:
        results = simulate_trajectory(
            initial_height=initial_height,
            m=m,
            u=u,
            surface_pressure_mb=surface_pressure_mb,
            surface_temp_celsius=surface_temp_celsius,
            v0=v0,
            CdA=CdA,
        )

        st.subheader("Results Overview")
        overview_cols = st.columns(3)
        overview_cols[0].metric(
            "Total Flight Time (s)", f"{results['total_time']:.2f}", help="Time until the UA reaches ground level"
        )
        overview_cols[1].metric(
            "Horizontal Distance (m)", f"{results['final_x']:.2f}", help="Total horizontal distance travelled"
        )
        overview_cols[2].metric(
            "Impact Angle (°)",
            f"{results['angle_from_vertical_deg']:.2f}",
            help="Angle between the UA velocity vector and vertical at impact",
        )

        dynamics_cols = st.columns(3)
        dynamics_cols[0].metric(
            "Final Vx (m/s)", f"{results['final_vx']:.2f}", help="Horizontal velocity due to wind"
        )
        dynamics_cols[1].metric(
            "Final Vz (m/s)", f"{results['final_vz']:.2f}", help="Vertical velocity at impact"
        )
        dynamics_cols[2].metric(
            "Resultant Speed (m/s)",
            f"{results['final_vxvz']:.2f}",
            help="Magnitude of the combined velocity components",
        )

        st.markdown("### Atmospheric Conditions")
        st.write(
            f"Temperature at {initial_height:.2f} m: **{results['initial_temperature']:.2f} K**"
        )
        st.write(
            f"Air density at {initial_height:.2f} m: **{results['initial_air_density']:.4f} kg/m³**"
        )

        st.markdown("### Ground Risk Zone Metrics")
        grz_data = pd.DataFrame(
            {
                "Metric": [
                    "GRZ Origin Position (m)",
                    "GRZ Radius (m)",
                    "GRZ Area (m²)",
                    "GRZ Minimum Distance (m)",
                    "GRZ Maximum Distance (m)",
                    "Probability of Collision (1 person)",
                ],
                "Value": [
                    f"{results['grz_origin']:.2f}",
                    f"{results['grz_radius']:.2f}",
                    f"{results['grz_area']:.2f}",
                    f"{results['grz_min']:.2f}",
                    f"{results['grz_max']:.2f}",
                    "Undefined" if math.isinf(results["p_coll"]) else f"{results['p_coll']:.5f}",
                ],
            }
        )
        st.dataframe(grz_data, use_container_width=True, hide_index=True)

        displacement_cols = st.columns(2)
        displacement_cols[0].metric(
            "Displacement from Initial Velocity (m)", f"{results['xvx']:.2f}"
        )
        displacement_cols[1].metric(
            "Displacement from Wind (m)", f"{results['final_x_wind']:.2f}"
        )

        st.subheader("Trajectory Visualisations")
        trajectory_chart = (
            alt.Chart(results["trajectory"])
            .mark_line(color="#4E79A7")
            .encode(
                x=alt.X("Time (s)", title="Time (s)"),
                y=alt.Y("Altitude (m)", title="Altitude (m)"),
            )
            .properties(height=320)
        )

        distance_chart = (
            alt.Chart(results["trajectory"])
            .mark_line(color="#F28E2B")
            .encode(
                x=alt.X("Horizontal Distance (m)", title="Horizontal Distance (m)"),
                y=alt.Y("Altitude (m)", title="Altitude (m)"),
            )
            .properties(height=320)
        )

        st.altair_chart(trajectory_chart, use_container_width=True)
        st.altair_chart(distance_chart, use_container_width=True)


if __name__ == "__main__":
    main()
