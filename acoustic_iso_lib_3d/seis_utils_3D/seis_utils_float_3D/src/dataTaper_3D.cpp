#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include "dataTaper_3D.h"
#include <math.h>

using namespace SEP;

// Constructor for offset only + end of trace taper
dataTaper_3D::dataTaper_3D(float maxOffset, float expOffset, float taperWidthOffset, std::string offsetMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry){

	// Data hypercube
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots -> consistent with number of shots from source and receiver geometry files
	_nReceiverPerShot = _dataHyper->getAxis(2).n; // Number of receiver per shot -> consistent with number of shots from source and receiver geometry files

	// Set flags
	_taperOffsetFlag = 1;
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	_taperEndTraceFlag = 1;
    _taperTimeFlag = 0;

	// Acquisition geometry files
	_sourceGeometry = sourceGeometry;
	_receiverGeometry = receiverGeometry;

	// Offset mask parameters
	_maxOffset=std::abs(maxOffset);
	_expOffset=expOffset;
	_taperWidthOffset=std::abs(taperWidthOffset);
	_offsetMuting=offsetMuting;

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

    computeTaperMaskOffset(); // Compute offset taper mask
	computeTaperEndTrace(); // Compute taper for end of trace
}

// Constructor for time only + end of trace
dataTaper_3D::dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, std::string timeMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry){

	// Data hypercube + parameters
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots -> consistent with number of shots from source and receiver geometry files
	_nReceiverPerShot = _dataHyper->getAxis(2).n; // Number of receiver per shot -> consistent with number of shots from source and receiver geometry files

	// Set flags
	_taperOffsetFlag = 0;
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	_taperEndTraceFlag = 1;
    _taperTimeFlag = 1;

	// Acquisition geometry files
	_sourceGeometry = sourceGeometry;
	_receiverGeometry = receiverGeometry;

    // Time mask parameters
	_t0=t0;
	_velMute=velMute;
	_expTime=expTime;
	_taperWidthTime=std::abs(taperWidthTime);
	_timeMuting=timeMuting;
	_moveout=moveout;

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

    computeTaperMaskTime(); // Compute offset taper mask
	computeTaperEndTrace(); // Compute taper for end of trace

}

// Constructor for offset + time + end of trace
dataTaper_3D::dataTaper_3D(float t0, float velMute, float expTime, float taperWidthTime, std::string moveout, std::string timeMuting, float maxOffset, float expOffset, float taperWidthOffset, std::string offsetMuting, float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper, std::shared_ptr<float2DReg> sourceGeometry, std::shared_ptr<float3DReg> receiverGeometry){

    // Display information about constructor
    std::cout << "---- [dataTaper_3D]: Data taper for time + offset + end of trace muting and tapering ----" << std::endl;

	// Data hypercube + parameters
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots -> consistent with number of shots from source and receiver geometry files
	_nReceiverPerShot = _dataHyper->getAxis(2).n; // Number of receiver per shot -> consistent with number of shots from source and receiver geometry files

	// Set flags
	_taperOffsetFlag = 1;
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]
	_taperEndTraceFlag = 1;
    _taperTimeFlag = 1;

	// Acquisition geometry files
	_sourceGeometry = sourceGeometry;
	_receiverGeometry = receiverGeometry;

    // Offset mask parameters
	_maxOffset=std::abs(maxOffset);
	_expOffset=expOffset;
	_taperWidthOffset=std::abs(taperWidthOffset);
	_offsetMuting=offsetMuting;

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording

    // Time mask parameters
	_t0=t0;
	_velMute=velMute;
	_expTime=expTime;
	_taperWidthTime=std::abs(taperWidthTime);
	_timeMuting=timeMuting;
	_moveout=moveout;

	// Compute masks
	computeTaperEndTrace(); // Compute taper for end of trace
	computeTaperMaskOffset(); // Compute offset taper mask
    computeTaperMaskTime(); // Compute offset taper mask

}

// Constructor for end of trace taper
dataTaper_3D::dataTaper_3D(float taperEndTraceWidth, std::shared_ptr<SEP::hypercube> dataHyper){

	// Data hypercube
	_dataHyper = dataHyper;
	_nShot = _dataHyper->getAxis(3).n; // Number of shots -> consistent with number of shots from source and receiver geometry files
	_nReceiverPerShot = _dataHyper->getAxis(2).n; // Number of receiver per shot -> consistent with number of shots from source and receiver geometry files

	// Set flags
	_taperOffsetFlag = 0;
	_taperEndTraceFlag = 1;
	_taperEndTraceWidth = taperEndTraceWidth; // Taper width [s]

    // Display information about constructor
	if (_taperEndTraceWidth > 0.0 ){std::cout << "---- [dataTaper_3D]: Data taper for end of trace tapering ----" << std::endl;}

	// Time sampling parameters
	_nts=_dataHyper->getAxis(1).n; // Number of time samples
	_ots=_dataHyper->getAxis(1).o; // Initial time
	_dts = _dataHyper->getAxis(1).d; // Time sampling
	_tMax = _ots+(_nts-1)*_dts; // Max time for recording
	computeTaperEndTrace(); // Compute taper for end of trace

}

// Compute offset mask
void dataTaper_3D::computeTaperMaskOffset(){

    // Display information about constructor
    if (_offsetMuting=="far"){
        std::cout << "---- [dataTaper_3D]: Data taper for far offset muting and end of trace tapering ----" << std::endl;
    } else {
        std::cout << "---- [dataTaper_3D]: Data taper for near offset muting and end of trace tapering ----" << std::endl;
    }

	// Allocate and computer taper mask
	_taperMaskOffset=std::make_shared<float2DReg>(_dataHyper->getAxis(2), _dataHyper->getAxis(3));
	_taperMaskOffset->set(1.0); // Set mask value to 1

    // Mute larger offsets
	if (_offsetMuting=="far") {

	   for (int iShot=0; iShot<_nShot; iShot++){

			// Compute shot position
			float sx=(*_sourceGeometry->_mat)[0][iShot];
			float sy=(*_sourceGeometry->_mat)[1][iShot];

			for (int iRec=0; iRec<_nReceiverPerShot;iRec++){

				// Compute horizontal position of receiver
				float rx=(*_receiverGeometry->_mat)[0][iRec][iShot];
				float ry=(*_receiverGeometry->_mat)[1][iRec][iShot];

				// Compute source/receiver distance
				float sourceRecDist = (sx-rx)*(sx-rx) + (sy-ry)*(sy-ry);
				sourceRecDist = std::sqrt(sourceRecDist);

				// Tapering zone
				if ( (_maxOffset <= sourceRecDist) && (sourceRecDist <= _maxOffset+_taperWidthOffset) ){

                    float argument = (sourceRecDist-_maxOffset)/_taperWidthOffset;
                    float weight=pow(cos(3.14159/2.0*argument), _expOffset);
                    (*_taperMaskOffset->_mat)[iShot][iRec] = weight;

				} else if (sourceRecDist >= _maxOffset+_taperWidthOffset){
					(*_taperMaskOffset->_mat)[iShot][iRec] = 0.0;
				} else {
                    // std::cout << "No tapering needed" << std::endl;
                }
			}
		}
	} else if (_offsetMuting=="near"){

        // Make sure taperWidth < maxOffset
        if (_taperWidthOffset > _maxOffset){
            std::cout << "**** ERROR [Offset muting]: Make sure taperWidthOffset < maxOffset ****" << std::endl;
            throw std::runtime_error("");
        }

        for (int iShot=0; iShot<_nShot; iShot++){

 			// Compute shot position
 			float sx=(*_sourceGeometry->_mat)[0][iShot];
 			float sy=(*_sourceGeometry->_mat)[1][iShot];

 			for (int iRec=0; iRec<_nReceiverPerShot;iRec++){

 				// Compute horizontal position of receiver
 				float rx=(*_receiverGeometry->_mat)[0][iRec][iShot];
 				float ry=(*_receiverGeometry->_mat)[1][iRec][iShot];

                // Compute source/receiver distance
 				float sourceRecDist = (sx-rx)*(sx-rx) + (sy-ry)*(sy-ry);
 				sourceRecDist = std::sqrt(sourceRecDist);

                // Intermediate zone
                if ( (sourceRecDist >= _maxOffset-_taperWidthOffset) && (sourceRecDist <= _maxOffset)) {
                    float argument = (_maxOffset-sourceRecDist)/_taperWidthOffset;
                    float weight=pow(cos(3.14159/2.0*argument), _expOffset);
                    (*_taperMaskOffset->_mat)[iShot][iRec] = weight;
                }
                // Zone outside (weight should be zero)
                else if (sourceRecDist <= _taperWidthOffset) {
                    (*_taperMaskOffset->_mat)[iShot][iRec] = 0.0;
                }
            }
        }
    } else {
		std::cout << "**** ERROR [dataTaper_3D]: Please select a value for 'offsetMuting' flag between 'near' and 'far' ****" << std::endl;
		throw std::runtime_error("");
	}
}

// Compute time mask
void dataTaper_3D::computeTaperMaskTime(){

    if (_timeMuting=="late"){
        std::cout << "---- [dataTaper_3D]: Data taper for late arrivals muting and end of trace tapering ----" << std::endl;
    } else {
        std::cout << "---- [dataTaper_3D]: Data taper for early arrivals muting and end of trace tapering ----" << std::endl;
    }

	// Allocate and computer taper mask
	_taperMaskTime=std::make_shared<float3DReg>(_dataHyper);
	_taperMaskTime->set(1.0); // Set mask value to 1

	// Mute late arrivals
	if (_timeMuting=="late") {

		for (int iShot=0; iShot<_nShot; iShot++){

            // Compute shot position
			float sx=(*_sourceGeometry->_mat)[0][iShot];
			float sy=(*_sourceGeometry->_mat)[1][iShot];

			// #pragma omp parallel for
			for (int iRec=0; iRec<_nReceiverPerShot;iRec++){

                // Compute horizontal position of receiver
				float rx=(*_receiverGeometry->_mat)[0][iRec][iShot];
				float ry=(*_receiverGeometry->_mat)[1][iRec][iShot];

				// Compute source/receiver distance
				float sourceRecDist = (sx-rx)*(sx-rx) + (sy-ry)*(sy-ry);
				sourceRecDist = std::sqrt(sourceRecDist);

				float tCutoff1, tCutoff2;
				int itCutoff1True, itCutoff2True, itCutoff1, itCutoff2;

				// Time cutoff #1
				if (_moveout=="linear"){
					// std::cout << "hyperbolic" << std::endl;
					tCutoff1=_t0+sourceRecDist/_velMute; // Compute linear time cutoff 1
					// std::cout << "iRec = " << iRec << std::endl;
					// std::cout << "tCutoff1 = " << tCutoff1 << std::endl;
				}
				if (_moveout=="hyperbolic" && _t0>=0){
					// std::cout << "hyperbolic" << std::endl;
					tCutoff1=std::sqrt(_t0*_t0+sourceRecDist*sourceRecDist/(_velMute*_velMute)); // Compute hyperbolic time cutoff 1
					// std::cout << "iRec = " << iRec << std::endl;
					// std::cout << "tCutoff1 = " << tCutoff1 << std::endl;
				}
				if (_moveout=="hyperbolic" && _t0<0){
					tCutoff1=std::sqrt(_t0*_t0+sourceRecDist*sourceRecDist/(_velMute*_velMute)) - 2.0*std::abs(_t0);
				}
				if (tCutoff1 < _ots){
					 itCutoff1True = (tCutoff1-_ots)/_dts-0.5; // Theoretical cutoff index (can be negative)
				} else {
					itCutoff1True = (tCutoff1-_ots)/_dts+0.5;
				}
				itCutoff1 = std::min(itCutoff1True, _nts-1);
				itCutoff1 = std::max(itCutoff1, 0);

				// Time cutoff #2
				tCutoff2=tCutoff1+_taperWidthTime;
				if (tCutoff2 < _ots){
					itCutoff2True = (tCutoff2-_ots)/_dts-0.5;
				} else {
					itCutoff2True = (tCutoff2-_ots)/_dts+0.5;
				}
				itCutoff2 = std::min(itCutoff2True, _nts-1);
				itCutoff2 = std::max(itCutoff2, 0);

				// Check the cutoff indices are different
				if (itCutoff2True == itCutoff1True){
					std::cout << "**** ERROR [Time muting]: Cutoff indices are identical. Use a larger taperWidth value ****" << std::endl;
					throw std::runtime_error("");
				}

				// Loop over time - Second zone where we taper the data
				for (int its=itCutoff1; its<itCutoff2; its++){
					float argument=1.0*(its-itCutoff1True)/(itCutoff2True-itCutoff1True);
					float weight=pow(cos(3.14159/2.0*argument), _expTime);
					(*_taperMaskTime->_mat)[iShot][iRec][its] = weight;
				}
				// Mute times after itCutoff2
				for (int its=itCutoff2; its<_nts; its++){
					(*_taperMaskTime->_mat)[iShot][iRec][its] = 0.0;
				}
			}
		}
	}

    // Mute early arrivals
    else if (_timeMuting=="early"){

        for (int iShot=0; iShot<_nShot; iShot++){

            // Compute shot position
			float sx=(*_sourceGeometry->_mat)[0][iShot];
			float sy=(*_sourceGeometry->_mat)[1][iShot];
            // std::cout << "iShot = " << iShot << std::endl;
            // std::cout << "sx = " << sx << std::endl;
            // std::cout << "sy = " << sy << std::endl;
            // std::cout << "sz = " << (*_sourceGeometry->_mat)[2][iShot] << std::endl;

			// #pragma omp parallel for
			for (int iRec=0; iRec<_nReceiverPerShot;iRec++){

                // Compute horizontal position of receiver
				float rx=(*_receiverGeometry->_mat)[0][iRec][iShot];
				float ry=(*_receiverGeometry->_mat)[1][iRec][iShot];
                // std::cout << "iRec = " << iRec << std::endl;
                // std::cout << "rx = " << rx << std::endl;
                // std::cout << "ry = " << ry << std::endl;
                // std::cout << "rz = " << (*_receiverGeometry->_mat)[2][iRec][iShot] << std::endl;

                // Compute source/receiver distance
				float sourceRecDist = (sx-rx)*(sx-rx) + (sy-ry)*(sy-ry);
				sourceRecDist = std::sqrt(sourceRecDist);

				float tCutoff1, tCutoff2;
				int itCutoff1True, itCutoff2True, itCutoff1, itCutoff2;

                // Time cutoff #1
				if (_moveout=="linear"){
					tCutoff1=_t0+sourceRecDist/_velMute; // Compute linear time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0>=0){
					tCutoff1=std::sqrt(_t0*_t0+sourceRecDist*sourceRecDist/(_velMute*_velMute)); // Compute hyperbolic time cutoff 1
				}
				if (_moveout=="hyperbolic" && _t0<0){
					tCutoff1=std::sqrt(_t0*_t0+sourceRecDist*sourceRecDist/(_velMute*_velMute)) - 2.0*std::abs(_t0);
				}
				if (tCutoff1 < _ots){
					 itCutoff1True = (tCutoff1-_ots)/_dts-0.5; // Theoretical cutoff index (can be negative)
				} else {
	 				itCutoff1True = (tCutoff1-_ots)/_dts+0.5;
				}
				itCutoff1 = std::min(itCutoff1True, _nts-1);
				itCutoff1 = std::max(itCutoff1, 0);
				// std::cout << "Cutoff 1 = " << itCutoff1 << std::endl;
				// Time cutoff #2
				tCutoff2=tCutoff1-_taperWidthTime;

				// Convert time cutoffs to index [sample]
				if (tCutoff2 < _ots){
					itCutoff2True = (tCutoff2-_ots)/_dts-0.5;
				} else {
					itCutoff2True = (tCutoff2-_ots)/_dts+0.5;
				}
				itCutoff2 = std::min(itCutoff2True, _nts-1);
				itCutoff2 = std::max(itCutoff2, 0);
				// std::cout << "Cutoff 2 = " << itCutoff2 << std::endl;
                // Loop over time - Mute the earlier times
				for (int its=0; its<itCutoff2; its++){
					(*_taperMaskTime->_mat)[iShot][iRec][its] = 0.0;
				}
				// Loop over time - Second zone where we taper the data
				for (int its=itCutoff2; its<itCutoff1; its++){
					float argument=1.0*(its-itCutoff2True)/(itCutoff1True-itCutoff2True);
					float weight=pow(sin(3.14159/2.0*argument), _expTime);
					(*_taperMaskTime->_mat)[iShot][iRec][its] = weight;
				}
            }
        }
    } else {
		std::cout << "**** ERROR [dataTaper_3D]: Please select a value for 'timeMuting' flag between 'early' and 'late' ****" << std::endl;
		throw std::runtime_error("");
	}
}

// Compute end of trace mask
void dataTaper_3D::computeTaperEndTrace(){

	// Allocate and computer taper mask
	_taperMaskEndTrace=std::make_shared<float1DReg>(_nts);
	_taperMaskEndTrace->set(1.0); // Set mask value to 1

	if (_taperEndTraceWidth > 0.0){

		// Time after which we start tapering the trace [s]
		float tTaperEndTrace = _tMax - _taperEndTraceWidth;

		// Make sure you're not out of bounds
		if (tTaperEndTrace < _ots){
			std::cout << "**** ERROR [End trace muting]: Make sure taperEndTraceWidth < total recording time ****" << std::endl;
			throw std::runtime_error("");
		}
		// Compute index from which you start tapering
		int itTaperEndTrace = (tTaperEndTrace-_ots)/_dts; // Index from which we start tapering
		// Compute trace taper
		for (int its=itTaperEndTrace; its<_nts; its++){
			float argument = 1.0*(its-itTaperEndTrace)/(_nts-1-itTaperEndTrace);
			(*_taperMaskEndTrace->_mat)[its] = pow(cos(3.14159/2.0*argument), 2);
		}
	}
}

// Forward
void dataTaper_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{

	if (!add) data->scale(0.0);

	// Offset only
	if (_taperOffsetFlag==1 && _taperTimeFlag==0){
		#pragma omp parallel for collapse(3)
		for (int iShot=0; iShot<_nShot; iShot++){
			for (int iRec=0; iRec<_nReceiverPerShot; iRec++){
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
					(*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*(*_taperMaskOffset->_mat)[iShot][iRec]*(*_taperMaskEndTrace->_mat)[its];
				}
			}
		}
	}

	// Time only
	else if (_taperOffsetFlag==0 && _taperTimeFlag==1){
        #pragma omp parallel for collapse(3)
		for (int iShot=0; iShot<_nShot; iShot++){
			for (int iRec=0; iRec<_nReceiverPerShot; iRec++){
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
                    (*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*(*_taperMaskEndTrace->_mat)[its]*(*_taperMaskTime->_mat)[iShot][iRec][its];
				}
			}
		}
    }

	// Offset and time
	else if (_taperOffsetFlag==1 && _taperTimeFlag==1){
        #pragma omp parallel for collapse(3)
		for (int iShot=0; iShot<_nShot; iShot++){
			for (int iRec=0; iRec<_nReceiverPerShot; iRec++){
				for (int its=0; its<_dataHyper->getAxis(1).n; its++){
                    (*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*(*_taperMaskOffset->_mat)[iShot][iRec]*(*_taperMaskEndTrace->_mat)[its]*(*_taperMaskTime->_mat)[iShot][iRec][its];
				}
			}
		}
    }

	else {
		// Only end of trace
		if (_taperEndTraceWidth > 0.0){
			#pragma omp parallel for collapse(3)
			for (int iShot=0; iShot<_nShot; iShot++){
				for (int iRec=0; iRec<_nReceiverPerShot; iRec++){
					for (int its=0; its<_dataHyper->getAxis(1).n; its++){
						(*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*(*_taperMaskEndTrace->_mat)[its];
					}
				}
			}
		}
		// Nothing
		else {
			data ->scaleAdd(model, 1.0, 1.0);
		}
	}
}

// Adjoint
void dataTaper_3D::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const{}
