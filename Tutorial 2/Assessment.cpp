#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	int user_input;
	int user_hist_input;
	int scan_input;
	string image_filename;
	int binSize;
	int num_of_bins;
	bool is_colour = false;;
	CImg<unsigned char> converted_image;
	cout << "Enter integer to represent input image:: \n1) Coloured Small Image\n2) Coloured Large Image\n3) Black and White Small Image\n4) Black and White Large Image\n";
	cin >> user_input;
	if (user_input == 1) image_filename = "test.ppm"; // colour image
	else if (user_input == 2 ) image_filename = "test_large.ppm"; // colour image larger
	else if (user_input == 3) image_filename = "test.pgm"; // black and white image
	else if (user_input == 4) image_filename = "test_large.pgm"; // black and white image larger


	cout << "Enter integer to represent initial histogram:: \n1) Simple Histogram \n2) Atomic Local Histogram\n";
	cin >> user_hist_input;

	cout << "Enter integer to represent scan:: \n1) Simple\n2) Blelloch Scan\n3) Hillis Steele\n";
	cin >> scan_input;

	cout << "Enter integer to represent number of Bins:: ";
	cin >> binSize;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str()); // a digital image in our project as a byte array containing pixels storing the intensity level for each colour channel
		// CImg stores pixels for each colour channel (RGB) separately, so the the pixels for red channel are stored first, followed by the green and then blue channel
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		std::vector<float> convolution_mask = {1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		if (image_input.spectrum() == 3) {
			is_colour = true; // Image is a coloured, 3 spectrum RGB image
			converted_image = image_input.get_RGBtoYCbCr();
			image_input = converted_image.get_channel(0);
		}

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/assessment_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		
		typedef int vectorType;
		// Declaring the initial Histogram, number of bins and their Bin Size which is determined by user input
		std::vector<vectorType> Histogram(binSize);
		size_t histogramSize = Histogram.size() * sizeof(vectorType);
		int min = 0;
		int max = 256;
		int binWidth = 256 / binSize;

		//Part 4 - device operations

		//device - buffers
		cl::Buffer buff_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer buff_histogram(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer buff_cumulHistogram(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer buff_LUT(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer buff_image_output(context, CL_MEM_READ_WRITE, image_input.size());

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(buff_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(buff_histogram, 0, 0, histogramSize);

		//4.2 Setup and execute the kernel (i.e. device code)
		// First kernel plotting a frequency histogram of each image pixel value
		cl::Kernel kernel_hist;

		// Allocating the function/kernel of the user specified histogram approach
		if (user_hist_input == 1) { // Simple Histogram
			kernel_hist = cl::Kernel(program, "hist_sim");
			kernel_hist.setArg(0, buff_image_input); // input image
			kernel_hist.setArg(1, buff_histogram); // output histogram
			kernel_hist.setArg(2, binWidth); // declaring the bin sizes / their width
		}
		else if (user_hist_input == 2) { // Atomic Local Histogram
			kernel_hist = cl::Kernel(program, "hist_loc_atom");
			kernel_hist.setArg(0, buff_image_input); // input image
			kernel_hist.setArg(1, buff_histogram); // output histogram
			kernel_hist.setArg(2, cl::Local(histogramSize)); // declaring a temporary histogram on local memory of the same size as the initial histogram
			kernel_hist.setArg(3, binWidth); // declaring the bin sizes / their width
		}

		cl::Event hist_event; // creating an event at the start of the kernel enqueue to track its runtime and memory transfers
		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_event); // enqueue the kernel parallel execution 
		queue.enqueueReadBuffer(buff_histogram, CL_TRUE, 0, histogramSize, &Histogram[0]);

		std::vector<vectorType> cumulHist(binSize);
		queue.enqueueFillBuffer(buff_cumulHistogram, 0, 0, histogramSize);

		// Second kernel plotting a cumulative histogram of every pixel in the image
		cl::Kernel kernel_cumulative_hist;
		
		// Allocating the function/kernel of the user specified scan approach
		if (scan_input == 1) {
			kernel_cumulative_hist = cl::Kernel(program, "cumul_hist");
		}
		else if (scan_input == 2) { // Blelloch
			kernel_cumulative_hist = cl::Kernel(program, "Blelloch_Scan");
		}
		else if (scan_input == 3) { // Hillis Steele
			kernel_cumulative_hist = cl::Kernel(program, "Hillis_Steele_Scan");
		}
		kernel_cumulative_hist.setArg(0, buff_histogram); // sets previous histogram as input
		kernel_cumulative_hist.setArg(1, buff_cumulHistogram); // stores result as cumulative histogram
		
		cl::Event cumul_hist_event; // creating an event at the start of the kernel enqueue to track its runtime and memory transfers
		queue.enqueueNDRangeKernel(kernel_cumulative_hist, cl::NullRange, cl::NDRange(cumulHist.size()), cl::NullRange, NULL, &cumul_hist_event); // enqueue the kernel parallel execution 
		queue.enqueueReadBuffer(buff_cumulHistogram, CL_TRUE, 0, histogramSize, &cumulHist[0]);

		std::vector<vectorType> LUT(binSize);
		queue.enqueueFillBuffer(buff_LUT, 0, 0, histogramSize);

		// Third kernel creates a look up table histogram of new pixel values by normalising the cumulative histogram values
		cl::Kernel kernel_LUT = cl::Kernel(program, "LUT_hist");
		kernel_LUT.setArg(0, buff_cumulHistogram);
		kernel_LUT.setArg(1, buff_LUT);
		kernel_LUT.setArg(2, binWidth);
		cl::Event hist_LUT_event;
		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(LUT.size()), cl::NullRange, NULL, &hist_LUT_event);
		queue.enqueueReadBuffer(buff_LUT, CL_TRUE, 0, histogramSize, &LUT[0]);

		// Fourth kernel adapts the LUT pixel values to the input image
		cl::Kernel kernel_edit = cl::Kernel(program, "edit");
		kernel_edit.setArg(0, buff_image_input);
		kernel_edit.setArg(1, buff_LUT);
		kernel_edit.setArg(2, buff_image_output);
		kernel_edit.setArg(3, binWidth);

		cl::Event hist_changes_event; // creating an event at the start of the kernel enqueue to track its runtime and memory transfers
		
		//4.3 Copy the result from device to host
		//Ouputting kernel execution and memory transfer times, and histogram values
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(kernel_edit, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_changes_event); // enqueue the kernel parallel execution 
		queue.enqueueReadBuffer(buff_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);
		
		// Histogram Output
		std::cout << "\n\nHistogram:::\n\n" << Histogram << std::endl; // outputting Histogram Values
		std::cout << "\nHistogram Kernel Execution Time (ns)::: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; // Outputting the calculated difference between event start and end to show its duration
		std::cout << GetFullProfilingInfo(hist_event, ProfilingResolution::PROF_US) << endl; // Outputting Full Event profiling information

		// Cumulative Histogram Output
		std::cout << "\n\nCumulative Histogram:::\n\n" << cumulHist << std::endl; // outputting Histogram Values
		std::cout << "Cumulative Histogram Kernel Execution Time (ns)::: " << cumul_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumul_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; // Outputting the calculated difference between event start and end to show its duration
		std::cout << GetFullProfilingInfo(cumul_hist_event, ProfilingResolution::PROF_US) << endl; // Outputting Full Event profiling information

		// Look Up Table Output
		std::cout << "\n\nLUT:::\n\n" << LUT << std::endl; // outputting Look Up Table Values
		std::cout << "LUT Kernel Execution Time (ns)::: " << hist_LUT_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_LUT_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; // Outputting the calculated difference between event start and end to show its duration
		std::cout << GetFullProfilingInfo(hist_LUT_event, ProfilingResolution::PROF_US) << endl; // Outputting Full Event profiling information

		// Final Image Adjustments Ouput
		std::cout << "Final Kernel Execution Time (ns)::: " << hist_changes_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_changes_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl; // Outputting the calculated difference between event start and end to show its duration
		std::cout << GetFullProfilingInfo(hist_changes_event, ProfilingResolution::PROF_US) << endl; // Outputting Full Event profiling information

		// Output
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum()); // Outputting the Final Image
		
		// Check to see if image to be output is RGB colour first
		if (is_colour == true) { // Checking if the original input image was in colour, and if it was, making sure to convert the final image bcak to colour as well
			for (int x = 0; x < converted_image.width(); x++) {
				for (int y = 0; y < converted_image.height(); y++) {
					converted_image(x, y, 0) = output_image(x, y);
				}
			}
			output_image = converted_image.get_YCbCrtoRGB();
		}

		CImgDisplay disp_output(output_image,"output"); // Output Final Image

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
