/** 
*  @file   larpandoracontent/LArWorkshop/TrainingExportAlgorithm.h 
* 
*  @brief  Header file for the TrainingExport algorithm class. 
* 
*  $Log: $ 
*/
#ifndef LAR_TrainingExport_ALGORITHM_H
#define LAR_TrainingExport_ALGORITHM_H 1
#include "Pandora/Algorithm.h"
#include "larpandoracontent/LArMonitoring/EventValidationAlgorithm.h"

//#ifdef MONITORING
//#include "PandoraMonitoringApi.h"
//#endif

#define IMSIZE 256 // Size of the generated image arrays 

namespace lar_content
{
/** 
*  @brief  TrainingExportAlgorithm class 
*/
class TrainingExportAlgorithm : public EventValidationAlgorithm
{

public:
/**     
*  @brief  Factory class for instantiating algorithm     
*/class Factory : public pandora::AlgorithmFactory    
{
public:        
pandora::Algorithm *CreateAlgorithm() const;    
};

private:    
pandora::StatusCode Run();    
pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
pandora::StatusCode PopulateImage(const pandora::CaloHitVector &caloHitVector, const float minX, const float minZ);
bool OneShowerMinBoundaries(const pandora::PfoList *const pPfoList, float &minX, float &minZ);
// Member variables here
//std::string m_pfoListName;
std::string m_caloHitListNameU;
std::string m_caloHitListNameV;
std::string m_caloHitListNameW;
};
//------------------------------------------------------------------------------------------------------------------------------------------
inline pandora::Algorithm *TrainingExportAlgorithm::Factory::CreateAlgorithm() const
{
return new TrainingExportAlgorithm();
}
} // namespace lar_content
#endif // #ifndef LAR_TrainingExport_ALGORITHM_H
