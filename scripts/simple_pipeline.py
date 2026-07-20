from arosics import COREG_LOCAL


def main():
    # Input images
    im_reference = "/home/bhaskar/Documents/ImageReg/Data/psdd_metop/metop/modis_1km.tif"
    im_target = "/home/bhaskar/Documents/ImageReg/2_outputs/05_avhrr_reflectance_ch2.tif"

    # AROSICS parameters
    kwargs = {
        "grid_res": 64,
        "window_size": (256, 256),
        "path_out": "auto",          # Automatically create output filename
        "projectDir": "my_project", 
        "min_reliability": 0.75, # Output directory
        "q":  True,
        "max_points":5000,
        "tieP_filter_level":3 # Show progress/messages
    }

    # Create the local co-registration object
    CRL = COREG_LOCAL(
        im_reference,
        im_target,
        **kwargs
    )

    # Perform local co-registration
    CRL.correct_shifts()

    print("Local image registration completed successfully.")


if __name__ == "__main__":
    main()




