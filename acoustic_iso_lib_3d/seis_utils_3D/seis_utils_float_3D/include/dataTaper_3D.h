#ifndef DATA_TAPER_3D_H
#define DATA_TAPER_3D_H 1

#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "operator.h"
#include <string>

using namespace SEP;

class dataTaper_3D : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

		float _maxOffset, _expOffset, _taperWidthOffset, _taperEndTraceWidth;
		float  _taperWidthTime, _expTime, _velMute, _t0;
		float _xMinRec, _xMaxRec, _dRec;
		float _xMinShot, _xMaxShot, _dShot;
		float _ots, _dts, _tMax;
		int _nts, _nReceiverPerShot, _nShot;
		std::string _moveout, _offsetMuting, _timeMuting;
		int _taperOffsetFlag, _taperEndTraceFlag, _taperTimeFlag; // Flags for type of muting

		// Arrays for tapering/muting
		std::shared_ptr<float2DReg> _taperMaskOffset;
		std::shared_ptr<float1DReg> _taperMaskEndTrace;
		std::shared_ptr<float3DReg> _taperMaskTime;

		// Data + acquisition geometry
		std::shared_ptr<float2DReg> _sourceGeometry;
		std::shared_ptr<float3DReg> _receiverGeometry;
		std::shared_ptr<SEP::hypercube> _dataHyper;

	public:

		// Constructor for offset tapering only
		dataTaper_3D(float maxOffset, float expOffset, float taperWidthOffset, std::string offsetMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry);

		// Constructor for time tapering only
		dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, std::string timeMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry);

		// Constructor for time and offset tapering
		dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, std::string timeMuting, float maxOffset, float expOffset, float taperWidthOffset, std::string offsetMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry);

		// Constructor for end of trace tapering only
		dataTaper_3D(float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper);

		/* Destructor */
		~dataTaper_3D(){};

		/* Mask computation */
		void computeTaperMaskOffset();
		void computeTaperMaskTime();
		void computeTaperEndTrace();

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Accessor */
		// Offset mask
		std::shared_ptr<float2DReg> getTaperMaskOffset_3D() {
			if (_taperOffsetFlag == 0) {
				std::cout << "**** ERROR [dataTaper_3D]: User requested tapering mask for offset which has not been used nor allocated (please set offset=1) ****" << std::endl;
				throw std::runtime_error("");
			}
			return _taperMaskOffset;
		}
		// Time mask
		std::shared_ptr<float3DReg> getTaperMaskTime_3D() {
			if (_taperTimeFlag == 0) {
				std::cout << "**** ERROR [dataTaper_3D]: User requested tapering mask for offset which has not been used nor allocated (please set offset=1) ****" << std::endl;
				throw std::runtime_error("");
			}
			return _taperMaskTime;
		}
		// End of trace mask
		std::shared_ptr<float1DReg> getTaperMaskEndTrace_3D() {
			if (_taperEndTraceFlag == 0) {
				std::cout << "**** ERROR [dataTaper_3D]: User requested for end of trace tapering mask which has not been used nor allocated (please set taperEndTraceWidth > 0.0) ****" << std::endl;
				throw std::runtime_error("");
			}
			return _taperMaskEndTrace;
		}
};

#endif
