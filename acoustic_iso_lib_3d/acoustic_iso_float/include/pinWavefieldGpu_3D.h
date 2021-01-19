#ifndef PIN_WAVEFIELD_GPU_3D_H
#define PIN_WAVEFIELD_GPU_3D_H 1

#include <vector>

using namespace SEP;

class pinWavefieldGpu_3D {

	public:

		/* Vectors containing wavefield addresses on pinned memory */
		std::vector<float*> _pinWavefieldVec1, _pinWavefieldVec2;

		/* Overloaded constructors */
		pinWavefieldGpu_3D(){
			std::cout << "Inside constructor pinWavefieldGpu_3D" << std::endl;
			_pinWavefieldVec1.clear();
			_pinWavefieldVec2.clear();
		}

		/* Destructor */
		~pinWavefieldGpu_3D(){};

};

#endif
