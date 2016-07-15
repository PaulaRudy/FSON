package cnnetwork;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import image.Align;

/**
 * This class is a simple demonstration of the FSON system.
 * 
 * @author Paula Rudy
 *
 */
public class FSONNetwork {
	
	public LinkedList<Layer> layers;
	public double[] out;
	
	public FSONNetwork() {

		Layer l1 = new Layer(76, 76, 3, 5, 5, 3, 32, 1, 0, LayerType.CONV);

		for (int i = 0; i < l1.K; i++) {
			double[][][] newFilter = new double[l1.Fdepth][l1.Frows][l1.Fcollumns];
			for (int x = 0; x < l1.Fdepth; x++) {
				for (int y = 0; y < l1.Frows; y++) {
					for (int z = 0; z < l1.Fcollumns; z++) {
						newFilter[x][y][z] = 1;
					}
				}
			}

			l1.filters.add(newFilter);
			double newBias = 1;
			l1.biases.add(newBias);
		}

		Layer l2 = new Layer(72, 72, 32, 3, 3, 32, 1, 2, 0, LayerType.MAXPOOL);
		Layer l3 = new Layer(35, 35, 32, 5, 5, 32, 16, 1, 0, LayerType.CONV);

		for (int i = 0; i < l3.K; i++) {
			double[][][] newFilter = new double[l3.Fdepth][l3.Frows][l3.Fcollumns];
			for (int x = 0; x < l3.Fdepth; x++) {
				for (int y = 0; y < l3.Frows; y++) {
					for (int z = 0; z < l3.Fcollumns; z++) {
						newFilter[x][y][z] = 0.5;
					}
				}
			}

			l3.filters.add(newFilter);
			double newBias = 1;
			l3.biases.add(newBias);
		}

		Layer l4 = new Layer(31, 31, 16, 5, 5, 16, 1, 2, 0, LayerType.MAXPOOL);
		Layer l5 = new Layer(14, 14, 16, 3, 3, 16, 16, 2, 0, LayerType.CONV);

		for (int i = 0; i < l5.K; i++) {
			double[][][] newFilter = new double[l5.Fdepth][l5.Frows][l5.Fcollumns];
			for (int x = 0; x < l5.Fdepth; x++) {
				for (int y = 0; y < l5.Frows; y++) {
					for (int z = 0; z < l5.Fcollumns; z++) {
						newFilter[x][y][z] = 1;
					}
				}
			}

			l5.filters.add(newFilter);
			double newBias = 1;
			l5.biases.add(newBias);
		}

		Layer l6 = new Layer(6, 6, 16, 3, 3, 1, 400, 1, 0, LayerType.LOCAL);

		for (int i = 0; i < l6.K; i++) {
			double[][][] newFilter = new double[l6.Fdepth][l6.Frows][l6.Fcollumns];
			for (int x = 0; x < l6.Fdepth; x++) {
				for (int y = 0; y < l6.Frows; y++) {
					for (int z = 0; z < l6.Fcollumns; z++) {
						newFilter[x][y][z] = 0.5;
					}
				}
			}

			l6.filters.add(newFilter);
			double newBias = 1;
			l6.biases.add(newBias);
		}

		Layer l7 = new Layer(5, 5, 16, 5, 5, 16, 2048, 1, 0, LayerType.FULLY);

		for (int i = 0; i < l7.K; i++) {
			double[][][] newFilter = new double[l7.Fdepth][l7.Frows][l7.Fcollumns];
			for (int x = 0; x < l7.Fdepth; x++) {
				for (int y = 0; y < l7.Frows; y++) {
					for (int z = 0; z < l7.Fcollumns; z++) {
						newFilter[x][y][z] = 1;
					}
				}
			}

			l7.filters.add(newFilter);
			double newBias = 1;
			l7.biases.add(newBias);
		}

		Layer l8 = new Layer(2048, 1, 1, 2048, 1, 1, 2016, 1, 0, LayerType.FULLY);

		for (int i = 0; i < l8.K; i++) {
			double[][][] newFilter = new double[l8.Fdepth][l8.Frows][l8.Fcollumns];
			for (int x = 0; x < l8.Fdepth; x++) {
				for (int y = 0; y < l8.Frows; y++) {
					for (int z = 0; z < l8.Fcollumns; z++) {
						newFilter[x][y][z] = 0.5;
					}
				}
			}

			l8.filters.add(newFilter);
			double newBias = 1;
			l8.biases.add(newBias);
		}

		this.out = new double[2016];
		
		this.layers = new LinkedList<Layer>();
		
		this.layers.add(0, l1);
		this.layers.add(1, l2);
		this.layers.add(2, l3);
		this.layers.add(3, l4);
		this.layers.add(4, l5);
		this.layers.add(5, l6);
		this.layers.add(6, l7);
		this.layers.add(7, l8);
		
	}
	
	public void calculate(BufferedImage image){

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat test = Align.bufferedImageToMat(image);

		Mat resizedImage = new Mat();
		Size sz = new Size(76, 76);
		Imgproc.resize(test, resizedImage, sz);

		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here in the order RGB
		Core.split(resizedImage, channels);
		
		for (int i = 0; i < channels.size(); i++) {

			channels.get(i).convertTo(channels.get(i), CvType.CV_64FC3);
			int size = (int) (channels.get(i).total() * channels.get(i).channels());
			double[] temp = new double[size];
			channels.get(i).get(0, 0, temp);
			for (int j = 0; j < 76; j++) {
				for (int k = 0; k < 76; k++) {
					this.layers.get(0).values[i][j][k] = temp[(j * 76) + k];
				}

			}

		}

		channels.get(0).convertTo(channels.get(0), CvType.CV_8UC1);
		double[] test2 = new double[(int) (channels.get(0).total() * channels.get(0).channels())];

		for (int j = 0; j < 76; j++) {
			for (int k = 0; k < 76; k++) {
				test2[(j * 76) + k] = this.layers.get(0).values[0][j][k];
			}

		}

		this.layers.get(0).convolution(this.layers.get(0).values, this.layers.get(0).filters, this.layers.get(1).values, this.layers.get(0).step, this.layers.get(0).pad, this.layers.get(0).biases);
		this.layers.get(1).pool(this.layers.get(1).values, this.layers.get(2).values, this.layers.get(1).step, this.layers.get(1).Fcollumns);
		this.layers.get(2).convolution(this.layers.get(2).values, this.layers.get(2).filters, this.layers.get(3).values, this.layers.get(2).step, this.layers.get(2).pad, this.layers.get(2).biases);
		this.layers.get(3).pool(this.layers.get(3).values, this.layers.get(4).values, this.layers.get(3).step, this.layers.get(3).Fcollumns);
		this.layers.get(4).convolution(this.layers.get(4).values, this.layers.get(4).filters, this.layers.get(5).values, this.layers.get(4).step, this.layers.get(4).pad, this.layers.get(4).biases);
		this.layers.get(5).local(this.layers.get(5).values, this.layers.get(5).filters, this.layers.get(6).values, this.layers.get(5).step, this.layers.get(5).pad, this.layers.get(5).biases);
		this.layers.get(6).full(this.layers.get(6).values, this.layers.get(6).filters, this.layers.get(7).values[0][0], this.layers.get(6).step, this.layers.get(6).pad, this.layers.get(6).biases);
		this.layers.get(7).full(this.layers.get(7).values, this.layers.get(7).filters, this.out, this.layers.get(7).step, this.layers.get(7).pad, this.layers.get(7).biases);

	}



	
}
