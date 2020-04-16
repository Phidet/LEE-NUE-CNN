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
int InteractionType(const pandora::MCParticleList *const pMCParticleList, const pandora::CaloHitList *const pCaloHitList, const pandora::PfoList *const pPfoList);
pandora::StatusCode PopulateImage(const pandora::CaloHitVector &caloHitVector, std::array<std::array<float,IMSIZE>,IMSIZE> &view, float (&label)[256][256][3]);
pandora::StatusCode MinBoundaries(const pandora::CaloHitVector &caloHitVector, float &minX, float &minZ);
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
