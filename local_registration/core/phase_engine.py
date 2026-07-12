import numpy as np

class PhaseCorrelationEngine:

    def __init__(self):

        self.eps = 1e-12

    def fft2(self, image):
        """
        Compute the 2-D Fast Fourier Transform.

        Parameters
        ----------
        image : ndarray

        Returns
        -------
        ndarray (complex)
        """

        return np.fft.fft2(image)


    def compute_cross_power_spectrum(
        self,
        reference_window,
        target_window
    ):
        """
        Compute the normalized cross-power spectrum.

        Foroosh (2002):

            Q = (Ft * Fr*) / |Ft * Fr*|

        where

            Ft  = FFT(Target)
            Fr  = FFT(Reference)

        Parameters
        ----------
        reference_window : ndarray

        target_window : ndarray

        Returns
        -------
        ndarray (complex)
        """

        # -----------------------------------------
        # FFT
        # -----------------------------------------

        # Replace NaN and Inf before FFT
        reference = np.nan_to_num(
            reference_window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        target = np.nan_to_num(
            target_window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # FFT

        F_ref = self.fft2(reference)

        F_tar = self.fft2(target)

        if not np.isfinite(F_ref).all():
            raise ValueError(
                "Invalid FFT in reference window."
            )

        if not np.isfinite(F_tar).all():
            raise ValueError(
                "Invalid FFT in target window."
            )

        # -----------------------------------------
        # Cross-power spectrum
        # -----------------------------------------

        cross_power = F_tar * np.conjugate(F_ref)

        # -----------------------------------------
        # Normalize magnitude
        # -----------------------------------------

        magnitude = np.abs(cross_power)

        magnitude[magnitude < self.eps] = self.eps

        cross_power = np.divide(
            cross_power,
            magnitude,
            out=np.zeros_like(cross_power),
            where=magnitude > self.eps
        )

        return cross_power


    def compute_correlation_surface(
        self,
        cross_power
    ):
        """
        Compute the correlation surface.

        Correlation Surface = IFFT(Cross-Power Spectrum)

        Parameters
        ----------
        cross_power : ndarray

        Returns
        -------
        ndarray (float32)
        """

        correlation_surface = np.fft.ifft2(
            cross_power
        )

        # Remove numerical imaginary component
        correlation_surface = np.real(
            correlation_surface
        )

        return correlation_surface.astype(
            np.float32
        )


    # def phase_correlate(
    #     self,
    #     reference_window,
    #     target_window
    # ):
    #     """
    #     Execute complete Step 16.

    #     Parameters
    #     ----------
    #     reference_window : ndarray

    #     target_window : ndarray

    #     Returns
    #     -------
    #     cross_power : ndarray

    #     correlation_surface : ndarray
    #     """

    #     cross_power = self.compute_cross_power_spectrum(

    #         reference_window,

    #         target_window

    #     )

    #     correlation_surface = self.compute_correlation_surface(

    #         cross_power

    #     )

    #     return cross_power, correlation_surface

    # def compute(
    #     self,
    #     reference_window,
    #     target_window
    # ):

    #     cross = self.compute_cross_power_spectrum(
    #         reference_window,
    #         target_window
    #     )

    #     surface = self.compute_correlation_surface(
    #         cross
    #     )

    #     return surface


    def compute(
        self,
        reference_window,
        target_window,
        return_cross_power=False
    ):

        cross = self.compute_cross_power_spectrum(
            reference_window,
            target_window
        )

        surface = self.compute_correlation_surface(
            cross
        )

        if return_cross_power:
            return cross, surface

        return surface