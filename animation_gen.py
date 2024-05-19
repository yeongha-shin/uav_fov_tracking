import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def create_animation(image_folder, output_file, frame_rate=2):
    # Get list of image files
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Create figure and axis
    fig, ax = plt.subplots()

    # List to store the images
    ims = []

    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img = plt.imread(img_path)
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=3000 / frame_rate, blit=True)

    # Save the animation
    ani.save(output_file, writer='ffmpeg')


# Example usage
if __name__ == "__main__":
    create_animation(image_folder='./output2/before/', output_file='./output2/ani_before.mp4', frame_rate=2)
