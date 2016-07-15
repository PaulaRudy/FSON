package testCNNetwork;

import static org.junit.Assert.*;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.junit.Before;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import cnnetwork.FSONNetwork;
import cnnetwork.Layer;
import cnnetwork.LayerType;
import image.Align;

public class TestCNNFSONnetwork {
	
	Layer l1, l2, l3, l4, l5, l6, l7, l8;
	double[] out;
	BufferedImage image;

	@Before
	public void setUp() throws Exception {
		// file named here must be in same location as Align class file
		image = ImageIO.read(Align.class.getResource("print.jpg"));

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		Mat test = Align.bufferedImageToMat(image);

		Mat resizedImage = new Mat();
		Size sz = new Size(76, 76);
		Imgproc.resize(test, resizedImage, sz);

		List<Mat> channels = new ArrayList<Mat>(3);// Channels are stored here in the order RGB
		Core.split(resizedImage, channels);

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

		out = new double[2016];

		for (int i = 0; i < channels.size(); i++) {

			channels.get(i).convertTo(channels.get(i), CvType.CV_64FC3);
			int size = (int) (channels.get(i).total() * channels.get(i).channels());
			double[] temp = new double[size];
			channels.get(i).get(0, 0, temp);
			for (int j = 0; j < 76; j++) {
				for (int k = 0; k < 76; k++) {
					l1.values[i][j][k] = temp[(j * 76) + k];
				}

			}

		}

		channels.get(0).convertTo(channels.get(0), CvType.CV_8UC1);
		double[] test2 = new double[(int) (channels.get(0).total() * channels.get(0).channels())];

		for (int j = 0; j < 76; j++) {
			for (int k = 0; k < 76; k++) {
				test2[(j * 76) + k] = l1.values[0][j][k];
			}

		}

		l1.convolution(l1.values, l1.filters, l2.values, l1.step, l1.pad, l1.biases);
		l2.pool(l2.values, l3.values, l2.step, l2.Fcollumns);
		l3.convolution(l3.values, l3.filters, l4.values, l3.step, l3.pad, l3.biases);
		l4.pool(l4.values, l5.values, l4.step, l4.Fcollumns);
		l5.convolution(l5.values, l5.filters, l6.values, l5.step, l5.pad, l5.biases);
		l6.local(l6.values, l6.filters, l7.values, l6.step, l6.pad, l6.biases);
		l7.full(l7.values, l7.filters, l8.values[0][0], l7.step, l7.pad, l7.biases);
		l8.full(l8.values, l8.filters, out, l8.step, l8.pad, l8.biases);
	}

	@Test
	public void test() {
		FSONNetwork test = new FSONNetwork();
		
		test.calculate(image);
		
		for (int i = 0; i < out.length; i++) {
			assertEquals(out[i], test.out[i], 0);
		}
	}

}
