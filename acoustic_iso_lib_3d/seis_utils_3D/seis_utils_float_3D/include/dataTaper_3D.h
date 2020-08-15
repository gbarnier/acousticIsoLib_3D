#ifndef DATA_TAPER_H
#define DATA_TAPER_H 1

#include "operator.h"
#include "float1DReg.h"
#include "float3DReg.h"
#include <string>

using namespace SEP;

class dataTaper_3D : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

		float _maxOffset, _expOffset, _taperWidthOffset, _taperEndTraceWidth;
		float  _taperWidthTime, _expTime, _velMute, _t0;
		float _xMinRec, _xMaxRec, _dRec;
		float _xMinShot, _xMaxShot, _dShot;
		float _ots, _dts, _tMax;
		int _nts, _nRec, _nShot;
		int _reverseTime, _reverseOffset;
		int _streamers;
		std::string _moveout;
		std::shared_ptr<float3DReg> _taperMask, _taperMaskTime, _taperMaskOffset;
		std::shared_ptr<SEP::hypercube> _dataHyper;
		std::shared_ptr<float1DReg> _taperEndTrace;

	public:

		// Constructor for both time and offset tapering
		dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, int reverseTime, float maxOffset, float expOffset, float taperWidthOffset, int reverseOffset, std::shared_ptr<SEP::hypercube> dataHyper, float taperEndTraceWidth, int streamers);

		// Constructor for time only tapering only
		dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::shared_ptr<SEP::hypercube> dataHyper, std::string moveout, int reverseTime, float taperEndTraceWidth);

		// Constructor for offset tapering only
		dataTaper_3D(float maxOffset, float _expOffset, float _taperWidthOffset, std::shared_ptr<SEP::hypercube> dataHyper, int _reverseOffset, float taperEndTraceWidth, int streamers);

		// Constructor for no tapering
		dataTaper_3D(std::shared_ptr<SEP::hypercube> dataHyper, float taperEndTraceWidth);

		/* Destructor */
		~dataTaper_3D(){};

		/* Masks computation */
		void computeTaperMaskTime();
		void computeTaperMaskOffset();
		void computeTaperMaskOffsetStreamers();
		void computeTaperEndTrace();

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Mutator */
		void setTaperMask_3D(std::shared_ptr<float3DReg> taperMask) {_taperMask = taperMask;}

		/* Accessor */
		std::shared_ptr<float3DReg> getTaperMask_3D() {return _taperMask;}
		std::shared_ptr<float3DReg> getTaperMaskTime_3D() {return _taperMaskTime;}
		std::shared_ptr<float3DReg> getTaperMaskOffset_3D() {return _taperMaskOffset;}
		std::shared_ptr<float1DReg> getTaperEndTrace_3D() {return _taperEndTrace;}

};

#endif
