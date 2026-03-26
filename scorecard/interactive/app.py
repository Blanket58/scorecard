from itertools import chain
from typing import List, Tuple, Union

import joblib
import numpy as np
import streamlit as st
from pydantic import BaseModel, field_validator

from scorecard.interactive.interactive import _load_data_raw


class _EnumBoundary(BaseModel):
    boundary: List[Tuple[Union[str, int, float], ...]]

    @field_validator("boundary")
    @classmethod
    def validate(cls, v):
        for group in v:
            if len(group) != 1:
                raise ValueError(
                    "Invalid group length: each tuple must have exactly one element."
                )

        items = list(chain.from_iterable(v))
        for item in items:
            is_valid = (
                isinstance(item, str)
                or isinstance(item, int)
                or isinstance(item, float)
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid value: {item} (type: {type(item).__name__}). "
                    "Only str / int / float / np.nan are allowed."
                )

        valid_items = list(
            filter(lambda item: not (isinstance(item, float) and np.isnan(item)), items)
        )
        if valid_items:
            if len(set([type(item) for item in valid_items])) != 1:
                raise ValueError("All non-NaN elements must have the same type.")

        if len(items) != len(set(items)):
            raise ValueError("Duplicate values not allowed.")

        converted_items = []
        for item in items:
            if isinstance(item, int):
                converted = np.int64(item)
            elif isinstance(item, float) and np.isnan(item) or isinstance(item, str):
                converted = item
            else:
                converted = np.float64(item)
            converted_items.append(converted)

        return converted_items


class _CharBoundary(BaseModel):
    boundary: List[Tuple[Union[str, float], ...]]

    @field_validator("boundary")
    @classmethod
    def validate(cls, v):
        for group in v:
            for item in group:
                is_str = isinstance(item, str)
                is_nan = isinstance(item, float) and np.isnan(item)
                if not (is_str or is_nan):
                    raise ValueError(
                        f"Invalid value: {item} (type: {type(item).__name__}). "
                        "Only str and np.nan are allowed in categorical boundaries."
                    )

        items = list(chain.from_iterable(v))
        if len(items) != len(set(items)):
            raise ValueError("Duplicate values not allowed.")
        return v


class _NumberBoundary(BaseModel):
    boundary: List[Union[int, float]]

    @field_validator("boundary")
    @classmethod
    def validate(cls, v):
        items = sorted(list(set(v)))
        converted_items = []
        for item in items:
            if np.isfinite(item):
                converted = np.float64(item)
            else:
                converted = item
            converted_items.append(converted)
        return converted_items


_boundary_strategy = {
    "Enum": _EnumBoundary,
    "Char": _CharBoundary,
    "Number": _NumberBoundary,
}


@st.cache_data
def _load_data():
    return _load_data_raw()


def main():
    self, X, y = _load_data()
    st.set_page_config(page_title="Binning Editor", layout="wide")
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("📊 WOE Binning Visualization & Editor")
    col_left, col_right = st.columns([0.3, 0.7])

    if "new_boundary" not in st.session_state:
        st.session_state["new_boundary"] = None
    if "new_bins_df" not in st.session_state:
        st.session_state["new_bins_df"] = None

    with col_left:
        feature_name = st.selectbox("Choose Feature", self.feature_names_in_)
        adjust_mode = st.toggle("Enable Binning Adjustment", value=False)
        if adjust_mode:
            st.divider()
            st.caption("Current boundary:")
            st.code(self.boundaries_[feature_name])
            new_boundary = st.text_input(
                label="Enter new boundary", placeholder="Must be valid python code"
            )
            if new_boundary:
                var_type = self.feature_types_[feature_name]
                validate_strategy = _boundary_strategy[var_type]
                _, calc_strategy = self._type_strategies[var_type]
                try:
                    new_boundary = eval(new_boundary, {"np": np, "__builtins__": {}})
                    new_boundary = validate_strategy(boundary=new_boundary).boundary
                    new_bins_df = calc_strategy(X[feature_name], y, new_boundary)
                    st.session_state.new_boundary = new_boundary
                    st.session_state.new_bins_df = new_bins_df
                    st.success(f"✅ New boundary: {new_boundary}")
                    st.divider()
                    save = st.button("Save (Can not be reversed.)", on_click=st.cache_data.clear)
                    if save:
                        self.boundaries_[feature_name] = new_boundary
                        self.bins_result_[feature_name] = new_bins_df
                        joblib.dump((self, X, y), 'model.pkl')
                        st.toast("Save success!")
                except Exception as e:
                    st.error(str(e))
        else:
            st.session_state.new_boundary = None
            st.session_state.new_bins_df = None

    with col_right:
        fig = self.plot(feature_name, figsize=(15, 6), return_fig=True)
        display_df = self.bins_result_[feature_name]
        if adjust_mode:
            if st.session_state.new_boundary:
                show_choice = st.segmented_control(
                    "Switch Plot", ("Original", "New"), default="New"
                )
            else:
                show_choice = None
            if show_choice == "New":
                fig = self._plot(
                    feature_name,
                    st.session_state.new_bins_df,
                    figsize=(15, 6),
                    return_fig=True,
                )
                display_df = st.session_state.new_bins_df
        st.pyplot(fig, clear_figure=True)
        st.dataframe(display_df, hide_index=True)


if __name__ == "__main__":
    main()
