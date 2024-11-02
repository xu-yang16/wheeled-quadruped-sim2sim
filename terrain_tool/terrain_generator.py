import os
import xml.dom.minidom
import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise

from math_utils import euler_to_quat, rot2d, rot3d, list_to_str

ROBOT = "wheeled_go1"
INPUT_SCENE_PATH = "./scene.xml"
OUTPUT_SCENE_PATH = f"../resources/{ROBOT}/xml/scene_terrain.xml"


class TerrainGenerator:
    def __init__(self) -> None:
        self.scene = xml_et.parse(
            os.path.join(os.path.dirname(__file__), INPUT_SCENE_PATH)
        )
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    # Add Box to scene
    def AddBox(
        self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[0.1, 0.1, 0.1]
    ):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size)
        )  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        geo.attrib["rgba"] = "0.92549 0.65882 0.5451 1"

    def AddGeometry(
        self,
        position=[1.0, 0.0, 0.0],
        euler=[0.0, 0.0, 0.0],
        size=[0.1, 0.1],
        geo_type="box",
    ):
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size)
        )  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(
        self,
        init_pos=[1.0, 0.0, 0.0],
        yaw=0.0,
        width=0.2,
        height=0.15,
        length=1.5,
        stair_nums=10,
    ):
        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2]],
                [0.0, 0.0, yaw],
                [width, length, height],
            )

    def AddSuspendStairs(
        self,
        init_pos=[1.0, 0.0, 0.0],
        yaw=1.0,
        width=0.2,
        height=0.15,
        length=1.5,
        gap=0.1,
        stair_nums=10,
    ):
        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2]],
                [0.0, 0.0, yaw],
                [width, length, abs(height - gap)],
            )

    def AddRoughGround(
        self,
        init_pos=[1.0, 0.0, 0.0],
        euler=[0.0, -0.0, 0.0],
        nums=[10, 10],
        box_size=[0.5, 0.5, 0.5],
        box_euler=[0.0, 0.0, 0.0],
        separation=[0.2, 0.2],
        box_size_rand=[0.05, 0.05, 0.05],
        box_euler_rand=[0.2, 0.2, 0.2],
        separation_rand=[0.05, 0.05],
    ):
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(
            separation_rand
        ) * np.random.uniform(-1.0, 1.0, 2)
        for i in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand
                ) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand
                ) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(
                    separation_rand
                ) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def AddPerlinHeighField(
        self,
        position=[1.0, 0.0, 0.0],  # position
        euler=[0.0, -0.0, 0.0],  # attitude
        size=[1.0, 1.0],  # width and length
        height_scale=0.2,  # max height
        negative_height=0.2,  # height in the negative direction of z axis
        image_width=128,  # height field image size
        img_height=128,
        smooth=100.0,  # smooth scale
        perlin_octaves=6,  # perlin noise parameter
        perlin_persistence=0.5,
        perlin_lacunarity=2.0,
        output_hfield_image="height_field.png",
    ):
        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                noise_value = noise.pnoise2(
                    x / smooth,
                    y / smooth,
                    octaves=perlin_octaves,
                    persistence=perlin_persistence,
                    lacunarity=perlin_lacunarity,
                )
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        dirname = os.path.dirname(
            os.path.join(os.path.dirname(__file__), OUTPUT_SCENE_PATH)
        )
        cv2.imwrite(os.path.join(dirname, output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height]
        )
        hfield.attrib["file"] = output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        geo.attrib["rgba"] = "0.498 0.7215 0.8705 1"

    def AddHeighFieldFromImage(
        self,
        position=[1.0, 0.0, 0.0],  # position
        euler=[0.0, -0.0, 0.0],  # attitude
        size=[2.0, 1.6],  # width and length
        height_scale=0.02,  # max height
        negative_height=0.1,  # height in the negative direction of z axis
        input_img=None,
        output_hfield_image="height_field.png",
        image_scale=[1.0, 1.0],  # reduce image resolution
        invert_gray=False,
    ):
        input_image = cv2.imread(
            os.path.join(os.path.dirname(__file__), input_img)
        )  # 替换为你的图像文件路径
        assert input_image is not None, "Image not found"
        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(
            input_image, (width, height), interpolation=cv2.INTER_AREA
        )
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - position

        dirname = os.path.dirname(
            os.path.join(os.path.dirname(__file__), OUTPUT_SCENE_PATH)
        )
        cv2.imwrite(os.path.join(dirname, output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height]
        )
        hfield.attrib["file"] = output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def Save(self):
        root_element = self.scene.getroot()
        raw_string = xml_et.tostring(root_element, "utf-8")
        parsed = xml.dom.minidom.parseString(raw_string)  # 解析字符串
        pretty_xml_as_string = parsed.toprettyxml()  # 美化格式
        with open(os.path.join(os.path.dirname(__file__), OUTPUT_SCENE_PATH), "w") as f:
            f.write(pretty_xml_as_string)  # Write to file


if __name__ == "__main__":
    tg = TerrainGenerator()

    # Box obstacle
    # tg.AddBox(position=[1.5, 0.0, 0.1], euler=[0, 0, 0.0], size=[1, 1.5, 0.2])

    # Geometry obstacle
    # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
    # tg.AddGeometry(
    #     position=[1.5, 0.0, 0.25],
    #     euler=[0, 0, 0.0],
    #     size=[1.0, 0.5, 0.5],
    #     geo_type="cylinder",
    # )

    # Slope
    # tg.AddBox(position=[2.0, 2.0, 0.5], euler=[0.0, -0.5, 0.0], size=[3, 1.5, 0.1])

    # Stairs
    # tg.AddStairs(init_pos=[1.0, 4.0, 0.0], yaw=0.0)

    # Suspend stairs
    # tg.AddSuspendStairs(init_pos=[1.0, 6.0, 0.0], yaw=0.0)

    # Rough ground
    tg.AddRoughGround(init_pos=[1, -1.0, 0.0], euler=[0, 0, 0.0], nums=[10, 8])
    # Heigh field from image
    tg.AddHeighFieldFromImage(
        position=[4, 0.0, 0.0],
        euler=[0, 0, 0],
        size=[2.0, 4.0],
        input_img="height_field.png",
        image_scale=[1.0, 1.0],
        output_hfield_image="height_field.png",
    )
    # Perlin heigh field
    tg.AddPerlinHeighField(position=[6, 0.0, 0.0], size=[2.0, 4], height_scale=0.15)

    tg.Save()
